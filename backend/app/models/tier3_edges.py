"""Edge construction for Tier-3 — what counts as a link, and what deliberately does not.

Kept separate from :mod:`app.models.tier3_graph` for the same reason
:mod:`app.models.tier1_features` is separate from the Tier-1 model and
:mod:`app.models.tier2_sequences` from the Tier-2 model: the ring algorithm is one piece of
code run over two corpora, and the only thing that differs between them is which edges exist.
Isolating that difference here is what keeps the claim "one algorithm, two real graphs"
checkable rather than aspirational.

--------------------------------------------------------------------------------------

**PaySim: the observed edge is not enough, measured rather than assumed.** On the train
split, 1,937,588 distinct origins carry 1,938,484 transactions — 99.95% of origins have
degree 1, the maximum is 2, and exactly 341 of 2,291,054 nodes are both an origin and a
destination. The observed money-flow graph is therefore a star forest: 353,807 disjoint
destination hubs with degree-1 leaves. Louvain over it returns the stars, which is
``groupby(counterparty_id)`` spelled expensively, and betweenness on a star restates degree.
Nothing multi-hop exists to detect.

The chain edge is what creates multi-hop structure, and it cannot be built on account names:
Phase 1 measured 0.00% of fraudulent transfers having a ``nameDest`` that reappears as any
fraudulent cash-out's ``nameOrig``. So a transfer is linked to a cash-out by *amount and step
proximity* (:func:`build_chain_edges`), and a name match is recorded as corroboration on the
edge rather than required for it.

**That edge rule is also a simulator artefact, and this module does not pretend otherwise.**
Exact-amount same-step matching selects 99.50% of fraudulent transfers against 0.23% of
legitimate ones, with a median of one candidate partner. It is PaySim's generative rule read
back out — the same species as Tier-1's PaySim PR-AUC of 0.9998 on ``amount ==
oldbalanceOrg``. The consequence is carried in the evaluation design, not here: a ring metric
laid naively over this edge measures the edge rule and not the graph, so ``train_tier3``
registers the rule as an explicit baseline and reports what topology adds *conditional on it
having already fired*. What this module owes that design is
:attr:`ChainEdges.candidate_count`, so ambiguity is a property of the edge rather than a
silent tie-break.

**IEEE-CIS: single columns are buckets, not identifiers.** ``card4`` and ``card6`` hold four
distinct values each, with 98,466 accounts on one of them; ``addr2`` holds 67 and
``P_emaildomain`` 59. Linking accounts on any of those produces one giant component and no
information. Only composite fingerprints survive: ``(card1, card2, card5, addr1)`` reaches
86.7% row coverage at a maximum of 427 accounts per fingerprint, and ``(DeviceInfo, id_30,
id_31, id_33)`` reaches 13.6% coverage with no column in common with the account UID.

**The circularity that has to be declared.** ``account_id`` for IEEE-CIS is constructed as
``c{card1}_a{addr1}_d{d1n}`` (:func:`app.data.adapters.build_ieee_cis_uid`). Two accounts
sharing the card fingerprint therefore differ *only* in ``d1n`` — which is either one card
fragmented into several inferred identities, or one card genuinely presenting as several
clients. Those are opposite findings and the graph cannot tell them apart, so every spec
records :attr:`FingerprintSpec.shared_with_uid` and ``train_tier3`` reports the device-only
graph alongside as the non-circular control.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Columns feeding :func:`app.data.adapters.build_ieee_cis_uid`. A fingerprint drawing on any
#: of them partly re-encodes the account identity it is supposed to be linking across, so the
#: overlap is recorded per spec rather than left for a reader to notice.
UID_COLUMNS: frozenset[str] = frozenset({"card1", "addr1", "D1", "P_emaildomain"})

#: PaySim transaction types, named here so the chain matcher does not carry string literals.
TRANSFER = "TRANSFER"
CASH_OUT = "CASH_OUT"

#: Beyond this many equally-plausible cash-out partners, a transfer is not identified — it is
#: merely common. The pairs are still emitted (dropping them would silently improve precision
#: by discarding the hard cases) but each carries ``1 / candidate_count`` as its weight, so an
#: ambiguous chain contributes proportionally less structure than an unambiguous one.
MAX_CHAIN_CANDIDATES = 8

#: An entity linking more than this many accounts is a bucket rather than an identifier and
#: creates no edges at all. This is the guard that stops the IEEE-CIS graph collapsing into a
#: single component; the value is selected on validation, and this is only the default.
DEFAULT_MAX_ENTITY_DEGREE = 200


@dataclass(frozen=True)
class FingerprintSpec:
    """One composite entity: the columns that identify it, and what it shares with the UID."""

    name: str
    columns: tuple[str, ...]

    @property
    def shared_with_uid(self) -> tuple[str, ...]:
        """Return this spec's columns that also feed the account UID construction."""
        return tuple(sorted(set(self.columns) & UID_COLUMNS))

    @property
    def is_circular(self) -> bool:
        """Return whether this fingerprint partly re-encodes the account identity."""
        return bool(self.shared_with_uid)


#: The IEEE-CIS entity set, chosen on measured cardinality and hubness rather than on which
#: columns sound like identifiers. Coverage and maximum accounts-per-entity on the train
#: split are quoted per spec; ``(addr1, P_emaildomain)`` was measured at 5,804 accounts on one
#: value and rejected as a hub before it reached this list.
IEEE_FINGERPRINTS: tuple[FingerprintSpec, ...] = (
    # 86.7% coverage, 33,253 distinct, max 427 accounts. Circular via card1 and addr1.
    FingerprintSpec("card_fp", ("card1", "card2", "card5", "addr1")),
    # 13.6% coverage, 3,689 distinct, max 1,991 accounts. Shares no column with the UID.
    FingerprintSpec("device_fp", ("DeviceInfo", "id_30", "id_31", "id_33")),
    # 30.9% coverage, 7,505 distinct, max 2,319 accounts. Circular via P_emaildomain only.
    FingerprintSpec("email_dist", ("P_emaildomain", "dist1")),
)

#: The subset sharing no column with the UID construction — the control graph.
NON_CIRCULAR_FINGERPRINTS: tuple[FingerprintSpec, ...] = tuple(
    spec for spec in IEEE_FINGERPRINTS if not spec.is_circular
)


@dataclass(frozen=True)
class ChainEdges:
    """Inferred transfer-to-cash-out links, with the ambiguity of each link attached.

    ``weight`` is ``1 / candidate_count``: a transfer matching one cash-out contributes a full
    edge, a transfer matching eight contributes an eighth of one to each. Ambiguity belongs on
    the edge because it is exactly what separates a mule chain from an amount coincidence, and
    a tie broken silently inside the matcher would delete that signal before the graph sees it.
    """

    frame: pd.DataFrame
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def matched_transfers(self) -> int:
        """Return how many distinct transfers found at least one partner."""
        if self.frame.empty:
            return 0
        return int(self.frame["transfer_txn"].nunique())


def _expand_ranges(
    begins: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Flatten per-row half-open index ranges into a pair of parallel index arrays.

    Given ``begins``/``ends`` describing one candidate range per transfer, returns
    ``(row_index, target_index)`` listing every ``(transfer, cash-out)`` combination the
    ranges cover. Written as an arange-minus-offset expansion rather than a Python loop
    because the matcher runs once per step, per step-offset, per hyperparameter candidate,
    and a per-pair interpreter loop dominates the whole phase's wall-clock.
    """
    counts = (ends - begins).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    row_index = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    target_index = np.repeat(begins, counts) + (np.arange(total, dtype=np.int64) - starts)
    return row_index, target_index


def build_chain_edges(
    frame: pd.DataFrame,
    *,
    step_window: int,
    amount_tolerance: float,
) -> ChainEdges:
    """Link each TRANSFER to the CASH_OUTs that plausibly drain the same mule.

    A transfer at step ``t`` for amount ``a`` is linked to every cash-out at step
    ``t' in [t, t + step_window]`` whose amount lies within ``a * (1 +/- amount_tolerance)``.
    The resulting edge joins the transfer's ``counterparty_id`` to the cash-out's
    ``account_id`` — the same mule wearing two labels, which is the only way to follow it
    given Phase 1's measured 0.00% name continuity.

    **Why looking forward in step is not leakage.** The match reads cash-outs later than the
    transfer, which would be a future read if this ran at scoring time. It does not: the frame
    handed in is always one rolling-window snapshot lying entirely in the past relative to the
    transaction being scored, and :class:`app.models.tier3_graph.EntityGraph` is what enforces
    that. Within a wholly-historical window, "later" is still history.

    Args:
        frame: Canonical PaySim rows for one window. Needs ``transaction_id``, ``account_id``,
            ``counterparty_id``, ``transaction_type``, ``amount`` and ``step``.
        step_window: How many steps after the transfer a cash-out may fall. Selected on
            validation.
        amount_tolerance: Relative amount tolerance, e.g. ``0.01`` for +/-1%. Selected on
            validation. Zero means exact equality.

    Returns:
        A :class:`ChainEdges` whose frame carries one row per candidate pair.

    Raises:
        ValueError: If ``step_window`` or ``amount_tolerance`` is negative.
    """
    if step_window < 0:
        raise ValueError(f"step_window must be non-negative, got {step_window}")
    if amount_tolerance < 0:
        raise ValueError(f"amount_tolerance must be non-negative, got {amount_tolerance}")

    empty = pd.DataFrame(
        {
            "transfer_txn": pd.Series(dtype="string"),
            "cashout_txn": pd.Series(dtype="string"),
            "mule_in": pd.Series(dtype="string"),
            "mule_out": pd.Series(dtype="string"),
            "step_gap": pd.Series(dtype="int16"),
            "amount_delta_rel": pd.Series(dtype="float64"),
            "candidate_count": pd.Series(dtype="int32"),
            "weight": pd.Series(dtype="float64"),
            "name_corroborated": pd.Series(dtype="bool"),
        }
    )

    types = frame["transaction_type"]
    transfers = frame.loc[types == TRANSFER]
    cashouts = frame.loc[types == CASH_OUT]
    if transfers.empty or cashouts.empty:
        return ChainEdges(
            empty,
            notes={"transfers": len(transfers), "cashouts": len(cashouts), "pairs": 0},
        )

    # Index cash-outs by step so the amount search only ever scans the eligible steps. Without
    # this the tolerance window would be searched against all 1.5M cash-outs at once, and at a
    # non-zero tolerance that is millions of spurious candidates rather than a bounded few.
    by_step: dict[int, dict[str, npt.NDArray[Any]]] = {}
    for step_value, group in cashouts.groupby("step", sort=False):
        amounts = group["amount"].to_numpy(dtype=np.float64)
        order = np.argsort(amounts, kind="stable")
        by_step[int(cast(int, step_value))] = {
            "amount": amounts[order],
            "txn": group["transaction_id"].to_numpy(dtype=object)[order],
            "account": group["account_id"].to_numpy(dtype=object)[order],
        }

    blocks: list[pd.DataFrame] = []
    for step_value, group in transfers.groupby("step", sort=False):
        amount = group["amount"].to_numpy(dtype=np.float64)
        low = amount * (1.0 - amount_tolerance)
        high = amount * (1.0 + amount_tolerance)
        txn = group["transaction_id"].to_numpy(dtype=object)
        dest = group["counterparty_id"].to_numpy(dtype=object)

        for offset in range(step_window + 1):
            bucket = by_step.get(int(cast(int, step_value)) + offset)
            if bucket is None:
                continue
            sorted_amounts = bucket["amount"]
            begins = np.searchsorted(sorted_amounts, low, side="left").astype(np.int64)
            ends = np.searchsorted(sorted_amounts, high, side="right").astype(np.int64)
            rows, targets = _expand_ranges(begins, ends)
            if rows.size == 0:
                continue
            source_amount = amount[rows]
            blocks.append(
                pd.DataFrame(
                    {
                        "transfer_txn": txn[rows],
                        "cashout_txn": bucket["txn"][targets],
                        "mule_in": dest[rows],
                        "mule_out": bucket["account"][targets],
                        "step_gap": np.full(rows.size, offset, dtype=np.int16),
                        "amount_delta_rel": np.divide(
                            np.abs(sorted_amounts[targets] - source_amount),
                            source_amount,
                            out=np.zeros(rows.size, dtype=np.float64),
                            where=source_amount != 0.0,
                        ),
                    }
                )
            )

    if not blocks:
        return ChainEdges(
            empty,
            notes={"transfers": len(transfers), "cashouts": len(cashouts), "pairs": 0},
        )

    pairs = pd.concat(blocks, ignore_index=True)
    counts = pairs.groupby("transfer_txn")["cashout_txn"].transform("size").to_numpy(dtype=np.int32)
    pairs["candidate_count"] = counts
    pairs["weight"] = 1.0 / np.maximum(counts, 1).astype(np.float64)
    pairs["name_corroborated"] = pairs["mule_in"].to_numpy(dtype=object) == pairs[
        "mule_out"
    ].to_numpy(dtype=object)
    for column in ("transfer_txn", "cashout_txn", "mule_in", "mule_out"):
        pairs[column] = pairs[column].astype("string")

    ambiguous = pairs["candidate_count"] > MAX_CHAIN_CANDIDATES
    notes = {
        "transfers": int(len(transfers)),
        "cashouts": int(len(cashouts)),
        "pairs": int(len(pairs)),
        "matched_transfers": int(pairs["transfer_txn"].nunique()),
        "median_candidates": float(
            pairs.groupby("transfer_txn")["candidate_count"].first().median()
        ),
        "ambiguous_pairs": int(ambiguous.sum()),
        "name_corroborated_pairs": int(pairs["name_corroborated"].sum()),
        "step_window": step_window,
        "amount_tolerance": amount_tolerance,
    }
    return ChainEdges(pairs.reset_index(drop=True), notes=notes)


def fingerprint_keys(frame: pd.DataFrame, spec: FingerprintSpec) -> "pd.Series[Any]":
    """Return the composite key per row, null where any component of the spec is missing.

    A fingerprint with a missing component is not a weaker fingerprint, it is a different one:
    treating absent ``id_33`` as a value of its own would merge every device that failed to
    report a screen resolution into a single entity, which is the exact hub collapse the
    composite is built to avoid.
    """
    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{spec.name} needs columns absent from the frame: {sorted(missing)}")

    present = frame[list(spec.columns)]
    complete = present.notna().all(axis=1)
    key = pd.Series(pd.NA, index=frame.index, dtype="string")
    if not complete.any():
        return key
    parts = [present.loc[complete, column].astype("string").fillna("") for column in spec.columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "|" + part
    key.loc[complete] = spec.name + ":" + joined
    return key


@dataclass(frozen=True)
class EntityEdges:
    """Bipartite account-to-entity incidences, with an IDF weight per entity.

    Bipartite rather than a projected account-to-account graph on purpose. Projecting an
    entity shared by ``k`` accounts emits ``k(k-1)/2`` pairs, so a single 427-account
    fingerprint becomes 91,000 edges and the tail dominates the graph's memory. The bipartite
    form is linear in the incidences, and its connected components are identical to the
    projection's — nothing the ring algorithm needs is lost.
    """

    frame: pd.DataFrame
    entity_degree: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def build_entity_edges(
    frame: pd.DataFrame,
    specs: Sequence[FingerprintSpec] = IEEE_FINGERPRINTS,
    *,
    max_entity_degree: int = DEFAULT_MAX_ENTITY_DEGREE,
) -> EntityEdges:
    """Link accounts to the composite entities they share, capping hubs and weighting by IDF.

    Two guards decide whether this graph carries information at all, and both are selected on
    validation rather than assumed:

    * **Hub cap.** An entity linking more than ``max_entity_degree`` accounts emits no edges.
      Without it ``card4``'s four values put 98,466 accounts in one component and every
      account's ring score becomes the same number.
    * **IDF weight.** ``log(total_accounts / accounts_sharing_entity)``, so a fingerprint two
      accounts share outweighs one four thousand share. Capping alone is a step function; the
      weight makes rarity continuous below the cap.

    Args:
        frame: Canonical IEEE-CIS rows for one window, carrying ``account_id`` and each spec's
            columns.
        specs: Which fingerprints to build. Defaults to :data:`IEEE_FINGERPRINTS`; pass
            :data:`NON_CIRCULAR_FINGERPRINTS` for the control graph.
        max_entity_degree: Accounts above which an entity is treated as a bucket.

    Returns:
        An :class:`EntityEdges` with one row per surviving account-entity incidence.

    Raises:
        ValueError: If ``max_entity_degree`` is below 2, which would emit no edges at all.
    """
    if max_entity_degree < 2:
        raise ValueError(f"max_entity_degree must be at least 2, got {max_entity_degree}")

    total_accounts = int(frame["account_id"].nunique()) or 1
    collected: list[pd.DataFrame] = []
    per_spec: dict[str, dict[str, Any]] = {}

    for spec in specs:
        keys = fingerprint_keys(frame, spec)
        usable = keys.notna()
        if not usable.any():
            per_spec[spec.name] = {"entities": 0, "edges": 0, "hubs_dropped": 0}
            continue

        incidences = pd.DataFrame(
            {"entity": keys.loc[usable], "account_id": frame.loc[usable, "account_id"]}
        ).drop_duplicates()

        degree = incidences.groupby("entity")["account_id"].nunique()
        # An entity one account holds links nothing; one above the cap links indiscriminately.
        keep = degree[(degree > 1) & (degree <= max_entity_degree)]
        hubs = int(((degree > max_entity_degree)).sum())
        if keep.empty:
            per_spec[spec.name] = {"entities": 0, "edges": 0, "hubs_dropped": hubs}
            continue

        kept = incidences.loc[incidences["entity"].isin(set(keep.index))].copy()
        shared = kept["entity"].map(keep).astype("float64")
        kept["weight"] = np.log(total_accounts / shared)
        kept["spec"] = spec.name
        kept["circular"] = spec.is_circular
        collected.append(kept)
        per_spec[spec.name] = {
            "entities": int(keep.size),
            "edges": int(len(kept)),
            "hubs_dropped": hubs,
            "max_degree": int(keep.max()),
            "mean_degree": float(keep.mean()),
            "circular_via": list(spec.shared_with_uid),
        }

    if not collected:
        empty = pd.DataFrame(
            {
                "entity": pd.Series(dtype="string"),
                "account_id": pd.Series(dtype="string"),
                "weight": pd.Series(dtype="float64"),
                "spec": pd.Series(dtype="string"),
                "circular": pd.Series(dtype="bool"),
            }
        )
        return EntityEdges(empty, notes={"per_spec": per_spec, "total_accounts": total_accounts})

    edges = pd.concat(collected, ignore_index=True)
    degrees = edges.groupby("entity")["account_id"].nunique().to_dict()
    notes = {
        "per_spec": per_spec,
        "total_accounts": total_accounts,
        "entities": int(edges["entity"].nunique()),
        "edges": int(len(edges)),
        "accounts_linked": int(edges["account_id"].nunique()),
        "max_entity_degree": max_entity_degree,
    }
    return EntityEdges(
        edges, entity_degree={str(k): int(v) for k, v in degrees.items()}, notes=notes
    )
