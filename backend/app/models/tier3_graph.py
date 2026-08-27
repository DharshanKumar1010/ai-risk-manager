"""Tier-3: transaction-network graph abuse-ring detection.

Phase 4 implements this layer: one ring-detection algorithm — Louvain community detection
plus degree and betweenness centrality — written against an abstract :class:`EntityGraph` and
instantiated over two real graphs, exposing

    flag_rings(graph_snapshot) -> list[RingFlag]

and, per transaction,

    Tier3Model.score(transaction) -> Tier3Result

The two graphs are :class:`PaySimMoneyFlowGraph`, where ring-level precision and recall can be
measured because PaySim is the only corpus carrying an observed money-flow edge, and
:class:`IEEECISSharedEntityGraph`, which produces the per-transaction ``ring_risk_score`` that
Phase 5's meta-learner and :class:`app.core.audit.AuditRecord` both reserve a slot for. One
algorithm, two real graphs, no transported score: a PaySim ring score is never attached to an
IEEE-CIS transaction, which :mod:`app.data.schema` states as a schema-level rule.

--------------------------------------------------------------------------------------

**Why this layer exists.** Tier-1 asks what one row says and Tier-2 asks what one account's
rhythm says. Neither can be asked whether *several* accounts are the same operation, because
neither can see more than one account at a time. Collusion is the failure mode that survives
both, and it is visible only in the topology.

**What the graph is, measured rather than assumed.** PaySim's observed money-flow graph is a
star forest — 99.95% of origins have degree 1 and 341 of 2,291,054 nodes are both an origin
and a destination — so Louvain over the observed edge alone returns ``groupby(nameDest)`` and
betweenness restates degree. Multi-hop structure exists only once the inferred transfer-to-
cash-out chain edge is added (:func:`app.models.tier3_edges.build_chain_edges`). That is the
whole reason this is a graph problem and not an aggregation.

**Three properties that decide whether the number means anything.**

*The snapshot is strictly historical.* A transaction at time ``t`` is scored against the most
recent snapshot whose ``snapshot_end <= t``, and its own edges are inserted only afterwards.
Edge construction inside a snapshot looks forward in step, which would be a future read at
scoring time and is not one here: the whole window already lies in the past. :meth:`Tier3Model
.score` cannot reach a graph that violates this because it never touches a graph at all — it
reads a score table the snapshot precomputed.

*The score reads topology, not money.* No feature in :data:`PAYSIM_RING_FEATURES` or
:data:`IEEE_RING_FEATURES` reads an amount or a balance. Amount enters only through *which
chain edges exist*, never as a magnitude — which matters because the chain rule alone
separates PaySim fraud at 99.50% against 0.23%, and a scorer allowed to read amounts would
simply relearn ``amount == oldbalanceOrg`` as Tier-1's PaySim model did. Keeping the scorer
structural is what makes "what does the graph add over the pairing rule" a question with an
answer.

*Accounts outside a ring abstain.* :class:`Tier3Result` carries ``ring_risk_score = None``,
never ``0.0``, for an account absent from the snapshot or sitting in a component below
:data:`MIN_RING_SIZE`. Emitting zero would tell Phase 5 "maximally clean" about an account
this layer has no opinion on, which is the rule Tier-1 states as *a missing feature is an
error, not a zero* and Tier-2 repeats for short sequences.
"""

import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.data.raw_spec import SourceDataset
from app.data.schema import TransactionFeatures
from app.ml.registry import artifact_path
from app.models.tier3_edges import (
    DEFAULT_MAX_ENTITY_DEGREE,
    IEEE_FINGERPRINTS,
    FingerprintSpec,
    build_chain_edges,
    build_entity_edges,
)

#: Communities smaller than this are not rings. Two accounts sharing one entity are a pair,
#: and calling a pair a ring would put most of the corpus in one, making the ring-level base
#: rate meaningless — precision without an interpretable base rate is not a result.
MIN_RING_SIZE = 3

#: A ring needs at least this many *junction* nodes — nodes of degree two or more. A single
#: hub surrounded by leaves is a star, and on PaySim a star is exactly one popular destination
#: account: 353,807 of them exist and none is a collusion structure. Without this filter the
#: detector "finds" every recurring destination and the ring-level metric measures destination
#: popularity. The criterion is deliberately structural — it names no chain edge, no amount
#: and no entity — so it excludes stars without smuggling the pairing rule into the definition
#: of what counts as a candidate.
MIN_BRANCH_NODES = 2

#: At or below this many accounts, a connected component is taken as one ring rather than
#: sub-partitioned. Louvain maximises modularity, which is not a meaningful objective on a
#: four-node path: measured on a 24-step PaySim window it split 121 of 121 chain-linked
#: components into halves that then fell below MIN_RING_SIZE, deleting precisely the structure
#: the chain edge exists to create. Above this size the partition earns its keep.
SMALL_COMPONENT_MAX = 12

#: Above this many nodes in one community, exact betweenness (O(nm)) is replaced by the
#: pivot-sampled estimator. Which one ran is recorded on every :class:`RingFlag`, because an
#: approximation silently substituted for an exact metric is not reportable.
BETWEENNESS_EXACT_MAX = 300

#: Pivot count for sampled betweenness above that size.
BETWEENNESS_PIVOTS = 64

#: Seed for Louvain (which is randomised) and for the sampled betweenness estimator. Logged
#: with every registry entry — ml-evaluation-standards section 5.
RANDOM_SEED = 42

#: p95 serving budget for one scoring call, in milliseconds. The same figure Tier-1 and
#: Tier-2 hold: all three sit behind the one Phase 7 endpoint and share a budget. Tier-3 meets
#: it by doing no graph work in the request path — see :meth:`Tier3Model.score`.
LATENCY_BUDGET_P95_MS = 50.0

#: Sequential calls in the latency benchmark, matching Tier-1's and Tier-2's.
LATENCY_BENCHMARK_CALLS = 100

#: Why an account was not scored. Recorded rather than folded into a score, so an audit row
#: says "Tier-3 had no opinion" instead of implying it had a favourable one.
ABSTAIN_NOT_IN_SNAPSHOT = (
    "account does not appear in the most recent graph snapshot; Tier-3 has seen no links for "
    "it and abstains rather than returning a score"
)
ABSTAIN_BELOW_MIN_RING = (
    f"account sits in a component smaller than {MIN_RING_SIZE} accounts; there is no ring to "
    "score and Tier-3 abstains rather than returning a score"
)

#: Structural features the PaySim scorer reads. No amount, no balance — see the module
#: docstring. ``ring_min_candidate_count`` is how unambiguous the ring's best chain link is,
#: which is a property of the match's uniqueness rather than of any sum of money.
PAYSIM_RING_FEATURES: tuple[str, ...] = (
    "ring_size",
    "ring_density",
    "ring_mean_degree",
    "ring_max_degree",
    "ring_chain_edge_count",
    "ring_chain_edge_share",
    "ring_min_candidate_count",
    "ring_name_corroborated",
    "account_degree",
    "account_degree_centrality",
    "account_betweenness",
    "account_chain_degree",
)

#: Structural features the IEEE-CIS scorer reads. Same rule: topology and entity rarity only.
IEEE_RING_FEATURES: tuple[str, ...] = (
    "ring_size",
    "ring_density",
    "ring_mean_degree",
    "ring_max_degree",
    "ring_mean_idf",
    "ring_spec_count",
    "ring_entity_count",
    "account_degree",
    "account_degree_centrality",
    "account_betweenness",
    "account_entity_count",
)

#: Columns no ring feature may derive from. Asserted by test, not promised here.
FORBIDDEN_FEATURE_SOURCES: frozenset[str] = frozenset(
    {
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "TransactionAmt",
        "amount_log",
        "is_fraud",
    }
)


def feature_names_for(source: SourceDataset) -> tuple[str, ...]:
    """Return the structural feature names the scorer reads for one corpus."""
    return PAYSIM_RING_FEATURES if source == "paysim" else IEEE_RING_FEATURES


@dataclass(frozen=True)
class GraphSnapshot:
    """One time-bounded graph, plus the provenance needed to report what it was built from.

    Frozen and self-describing on purpose: a ring metric is only interpretable next to the
    window it was measured over, and bundling the two means a caller cannot report the flag
    without the window.
    """

    graph: "nx.Graph[str]"
    source_dataset: SourceDataset
    window_start: datetime
    snapshot_end: datetime
    is_bipartite: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def account_nodes(self) -> list[str]:
        """Return the account-kind nodes, excluding the entity nodes of a bipartite graph."""
        return [
            str(node)
            for node in self.graph.nodes
            if self.graph.nodes[node].get("kind", "account") == "account"
        ]


class EntityGraph(ABC):
    """A time-bounded account graph. Subclasses supply edges; the ring algorithm is shared.

    The buffer/insert/evict shape is deliberate even though Phase 4 rebuilds each snapshot
    from scratch. The phase brief calls an incremental rolling-window graph the highest-
    leverage enhancement in the project, and :meth:`insert`, :meth:`evict_before` and
    :meth:`dirty_accounts` are the seams it needs: swapping a full rebuild for a re-run over
    touched components becomes a change inside :meth:`snapshot` rather than a rewrite of every
    caller. Holding the library graph behind this wrapper is the same convention
    ``pyproject.toml`` states for ``Tier1Model`` and ``CostModel``, and it is also what makes
    a NetworkX-to-igraph swap a backend decision rather than an API break.
    """

    source_dataset: SourceDataset

    def __init__(self) -> None:
        self._buffer: list[pd.DataFrame] = []
        self._dirty: set[str] = set()

    def insert(self, frame: pd.DataFrame) -> None:
        """Add transactions to the window. Does not build anything until :meth:`snapshot`."""
        if frame.empty:
            return
        self._buffer.append(frame)
        self._dirty.update(str(value) for value in frame["account_id"].dropna().unique())

    def evict_before(self, cutoff: datetime) -> int:
        """Drop buffered transactions strictly older than ``cutoff``.

        Returns:
            How many rows were evicted. Reported rather than discarded, because a window that
            silently keeps everything and one that correctly expires look identical from the
            outside until a metric moves.
        """
        if not self._buffer:
            return 0
        combined = pd.concat(self._buffer, ignore_index=True)
        keep = combined["event_time"] >= cutoff
        evicted = int((~keep).sum())
        self._buffer = [combined.loc[keep].reset_index(drop=True)]
        return evicted

    def dirty_accounts(self) -> set[str]:
        """Return accounts touched since the last snapshot — the incremental-update seam."""
        return set(self._dirty)

    def buffered(self) -> pd.DataFrame:
        """Return everything currently in the window as one frame."""
        if not self._buffer:
            return pd.DataFrame()
        if len(self._buffer) > 1:
            self._buffer = [pd.concat(self._buffer, ignore_index=True)]
        return self._buffer[0]

    def snapshot(self, end: datetime) -> GraphSnapshot:
        """Build the graph over everything currently buffered.

        Args:
            end: The snapshot's end time. Every buffered row must precede it — a row at or
                after ``end`` means a caller inserted before scoring, which is the leak this
                layer's whole time discipline exists to prevent.

        Raises:
            ValueError: If any buffered transaction is not strictly earlier than ``end``.
        """
        frame = self.buffered()
        if not frame.empty:
            latest = frame["event_time"].max()
            if latest >= end:
                raise ValueError(
                    f"snapshot at {end.isoformat()} contains a transaction at "
                    f"{latest.isoformat()}; a snapshot may only be built from strictly "
                    "earlier transactions"
                )
        window_start = frame["event_time"].min() if not frame.empty else end
        graph, notes = self._build(frame)
        self._dirty.clear()
        return GraphSnapshot(
            graph=graph,
            source_dataset=self.source_dataset,
            window_start=cast(datetime, window_start),
            snapshot_end=end,
            is_bipartite=self.is_bipartite,
            notes=notes,
        )

    #: Whether nodes are accounts and entities (True) or accounts alone (False). This decides
    #: whether the star filter applies — see :func:`detect_communities`.
    is_bipartite: bool = False

    @abstractmethod
    def _build(self, frame: pd.DataFrame) -> "tuple[nx.Graph[str], dict[str, Any]]":
        """Return the graph for one window, plus provenance notes."""


class PaySimMoneyFlowGraph(EntityGraph):
    """Observed money-flow edges plus inferred transfer-to-cash-out chain edges.

    Two relations, carried as independent ``flow`` and ``chain`` boolean flags on each edge so
    their contributions stay separable *and* a pair joined by both keeps both. Flags rather
    than one mutually exclusive label because a mule is routinely a flow endpoint and a chain
    endpoint at once, and collapsing that to a single label deletes the observed edge:

    * ``flow`` — the observed ``account_id -> counterparty_id`` edge. On its own this is a
      star forest and carries no ring structure, but it supplies the victim-to-mule and
      mule-to-destination legs the chain edge stitches together.
    * ``chain`` — the inferred link joining a transfer's destination to a cash-out's origin,
      matched on amount and step because Phase 1 measured account-name continuity at 0.00%.

    Only ``chain`` creates paths longer than two hops, so only ``chain`` makes this a graph.
    """

    source_dataset: SourceDataset = "paysim"

    def __init__(self, *, step_window: int, amount_tolerance: float) -> None:
        super().__init__()
        self.step_window = step_window
        self.amount_tolerance = amount_tolerance

    def _build(self, frame: pd.DataFrame) -> "tuple[nx.Graph[str], dict[str, Any]]":
        graph: nx.Graph[str] = nx.Graph()
        if frame.empty:
            return graph, {"flow_edges": 0, "chain_edges": 0}

        flow = frame[["account_id", "counterparty_id"]].dropna()
        for origin, destination in zip(
            flow["account_id"].to_numpy(dtype=object),
            flow["counterparty_id"].to_numpy(dtype=object),
            strict=True,
        ):
            left, right = str(origin), str(destination)
            if graph.has_edge(left, right):
                graph.edges[left, right]["flow"] = True
                continue
            graph.add_edge(
                left,
                right,
                flow=True,
                chain=False,
                weight=1.0,
                candidate_count=0,
                name_corroborated=False,
            )

        chain = build_chain_edges(
            frame, step_window=self.step_window, amount_tolerance=self.amount_tolerance
        )
        pairs = chain.frame
        for left_raw, right_raw, weight, candidates, corroborated in zip(
            pairs["mule_in"].to_numpy(dtype=object),
            pairs["mule_out"].to_numpy(dtype=object),
            pairs["weight"].to_numpy(dtype=np.float64),
            pairs["candidate_count"].to_numpy(dtype=np.int64),
            pairs["name_corroborated"].to_numpy(dtype=bool),
            strict=True,
        ):
            left, right = str(left_raw), str(right_raw)
            if left == right:
                # A name match is corroboration, never a link in its own right: a self-loop
                # adds no path and would inflate degree for the one case Phase 1 measured at
                # essentially zero frequency.
                continue
            if graph.has_edge(left, right):
                existing = graph.edges[left, right]
                # Accumulate onto whatever is already there rather than replacing it. An
                # earlier version overwrote the attribute dict, which silently deleted the
                # observed flow edge whenever a mule was also a chain endpoint -- inflating
                # `ring_chain_edge_count` and `account_chain_degree`, both scorer features.
                if existing.get("chain"):
                    existing["weight"] = float(existing["weight"]) + float(weight)
                    existing["candidate_count"] = min(
                        int(existing["candidate_count"]), int(candidates)
                    )
                else:
                    existing["chain"] = True
                    existing["weight"] = float(weight)
                    existing["candidate_count"] = int(candidates)
                existing["name_corroborated"] = bool(existing.get("name_corroborated")) or bool(
                    corroborated
                )
                continue
            graph.add_edge(
                left,
                right,
                flow=False,
                chain=True,
                weight=float(weight),
                candidate_count=int(candidates),
                name_corroborated=bool(corroborated),
            )

        # Every node on this graph is an account -- PaySim has no entity nodes. Set it
        # explicitly so ``detect_communities`` and the feature walk can read one attribute
        # uniformly across both graphs rather than branching on which corpus they are on.
        for node in graph:
            graph.nodes[node]["kind"] = "account"
        notes = {
            "flow_edges": int(len(flow)),
            "chain": chain.notes,
            "step_window": self.step_window,
            "amount_tolerance": self.amount_tolerance,
        }
        return graph, notes


class IEEECISSharedEntityGraph(EntityGraph):
    """Bipartite account-to-entity graph over composite fingerprints.

    Bipartite rather than projected: an entity shared by ``k`` accounts would emit
    ``k(k-1)/2`` account pairs, so one 427-account fingerprint becomes 91,000 edges. The
    bipartite form is linear in the incidences and has identical connected components.

    Entity nodes carry ``kind="entity"`` so ring size counts accounts only — a ring of three
    accounts joined through two devices is a ring of three, not of five.

    The star filter that applies to PaySim is deliberately *not* applied here, which
    :attr:`is_bipartite` selects. On PaySim a star's hub is itself an account, so the star is
    one popular destination and not a ring. Here the hub is an entity, and several accounts on
    one device is precisely the structure this graph exists to surface. The equivalent guard
    against an over-popular hub is the entity degree cap in
    :func:`app.models.tier3_edges.build_entity_edges`, applied when the edges are built rather
    than when the communities are read.
    """

    source_dataset: SourceDataset = "ieee_cis"
    is_bipartite = True

    def __init__(
        self,
        *,
        specs: Sequence[FingerprintSpec] = IEEE_FINGERPRINTS,
        max_entity_degree: int = DEFAULT_MAX_ENTITY_DEGREE,
    ) -> None:
        super().__init__()
        self.specs = tuple(specs)
        self.max_entity_degree = max_entity_degree

    def _build(self, frame: pd.DataFrame) -> "tuple[nx.Graph[str], dict[str, Any]]":
        graph: nx.Graph[str] = nx.Graph()
        if frame.empty:
            return graph, {"edges": 0}

        edges = build_entity_edges(frame, self.specs, max_entity_degree=self.max_entity_degree)
        for account_raw, entity_raw, weight, spec, circular in zip(
            edges.frame["account_id"].to_numpy(dtype=object),
            edges.frame["entity"].to_numpy(dtype=object),
            edges.frame["weight"].to_numpy(dtype=np.float64),
            edges.frame["spec"].to_numpy(dtype=object),
            edges.frame["circular"].to_numpy(dtype=bool),
            strict=True,
        ):
            account, entity = str(account_raw), str(entity_raw)
            graph.add_node(account, kind="account")
            graph.add_node(entity, kind="entity", spec=str(spec), circular=bool(circular))
            graph.add_edge(
                account, entity, flow=False, chain=False, entity=True, weight=float(weight)
            )
        return graph, dict(edges.notes)


class RingFlag(BaseModel):
    """One detected community, the metrics that describe it, and its risk score.

    Carries the member account ids and the centrality metrics that drove the flag, per the
    Phase 4 brief. ``betweenness_exact`` records whether the estimator was sampled, because a
    reader comparing two rings needs to know they were measured the same way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ring_id: str = Field(min_length=1, description="Stable within one snapshot.")
    member_account_ids: tuple[str, ...] = Field(description="Account nodes only.")
    member_scores: tuple[float, ...] = Field(
        default=(),
        description="Each member's own score, aligned to member_account_ids. Carried so the "
        "served per-account table is built from the same numbers the offline metric uses; "
        "without it a member inherits the ring's maximum and the two views disagree.",
    )
    size: int = Field(ge=0, description="Member account count.")
    ring_risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum member score. None when no scorer was fitted, which is the "
        "structural-only path used to build training data.",
    )
    density: float = Field(ge=0.0, description="Realised edges over possible edges.")
    max_degree_centrality: float = Field(
        ge=0.0,
        description="Highest member degree divided by ring size minus one. On the bipartite "
        "IEEE-CIS graph a member's degree counts the entity nodes it sits on, so this can "
        "exceed 1.0 for an account carrying more shared entities than the ring has other "
        "accounts. That is meaningful rather than a defect -- it is the account that ties the "
        "ring together -- but it is not a fraction, and is not to be rendered as a percentage.",
    )
    max_betweenness: float = Field(ge=0.0)
    betweenness_exact: bool = Field(
        description="False when the pivot-sampled estimator was used above "
        f"{BETWEENNESS_EXACT_MAX} nodes.",
    )
    chain_edge_count: int = Field(ge=0, description="PaySim inferred links. 0 elsewhere.")
    entity_count: int = Field(ge=0, description="IEEE-CIS shared entities. 0 elsewhere.")
    snapshot_end: datetime
    drivers: tuple[str, ...] = Field(
        default=(),
        description="Which metrics put this ring above threshold, for the Phase 8 panel.",
    )


class RingGraphNode(BaseModel):
    """One node of a flagged ring's topology, as exported for the Phase 8 network view.

    An account node's ``node_id`` is the plain ``account_id`` -- already exposed by
    ``RingResponse.members``, so hashing it here would just be a second encoding of a value the
    same response already carries in the clear. An entity node's ``node_id`` is never the raw
    composite fingerprint (a ``card1``/``card4``/... tuple, or worse a device fingerprint):
    :func:`export_ring_edges` hashes it, because the fingerprint itself is exactly the kind of
    device/card signal :mod:`app.models.tier3_edges` builds from raw identity columns that no
    other response on this API returns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    kind: str = Field(description="'account' or 'entity'.")
    entity_type: str | None = Field(
        default=None,
        description="The fingerprint spec name (e.g. 'card1_card4'). None for account nodes.",
    )


class RingGraphEdge(BaseModel):
    """One edge of a flagged ring's topology: an account-account or account-entity incidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(description="A RingGraphNode.node_id.")
    target: str = Field(description="A RingGraphNode.node_id.")


def _anonymized_entity_id(entity: str, key: bytes) -> str:
    """Hash a raw entity fingerprint down to an opaque node id, keyed so it cannot be reversed.

    **Keyed, not merely hashed -- found in security review, not designed in up front.** An
    IEEE-CIS shared-entity fingerprint (``card_fp:13926|321.0|226.0|299.0``, say) is built from
    a handful of columns each drawn from a small, public domain: ``card1`` has on the order of
    13,000 distinct values in the corpus, ``card2``/``card5``/``addr1`` far fewer, and the
    device columns (``DeviceInfo``, ``id_30``, ``id_31``, ``id_33``) are in the low hundreds
    each. That is a candidate space in the thousands to low millions -- trivial to enumerate
    offline against a plain ``sha256(entity)``, so an unkeyed hash is not an anonymization at
    all, just a fixed-width re-encoding a dictionary attack undoes in seconds. Truncating to 16
    hex characters helps against a *collision* (two distinct entities landing on the same id,
    genuinely unlikely at these ring sizes) but does nothing against a *preimage* search over a
    known, small candidate set, which is the actual threat here.

    ``key`` -- ``Settings.entity_anonymization_key``, deployment-specific and never persisted
    into a :class:`Tier3Model` artifact or ``models/registry.json`` -- is what makes the
    mapping non-reversible without it: HMAC-SHA256 is a pseudorandom function, so a candidate
    fingerprint's hash no longer matches its plaintext hash from outside this process, and only
    someone holding the key can reproduce the mapping either forward (to check a guess) or
    build the dictionary in the first place. Still 16 hex characters, for the same
    collision-resistance reasoning as before.
    """
    return hmac.new(key, entity.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def export_ring_edges(
    snapshot: GraphSnapshot, communities: Sequence[Sequence[str]], *, key: bytes
) -> dict[str, tuple[tuple[RingGraphNode, ...], tuple[RingGraphEdge, ...]]]:
    """Return each community's topology as anonymized nodes and edges, keyed by ring id.

    Ring ids match :func:`ring_feature_frame`'s ``f"r{index}"`` over the same ``communities``
    list -- both are a plain ``enumerate``, so calling this with the exact communities
    :func:`detect_communities` produced for one snapshot reproduces the same ids that
    snapshot's served :class:`Tier3Model` uses for ``ring_of``/``ring_sizes``.

    Walks each community the same way :func:`ring_feature_frame` does: an account-account edge
    within the community (PaySim's inferred chain links), or an account-entity edge onto a
    shared fingerprint (IEEE-CIS's bipartite graph). Only edges with both endpoints inside the
    community are kept -- an entity shared with an account outside this ring is real structure
    but belongs to a different ring's picture, not this one's.

    ``key`` is ``Settings.entity_anonymization_key`` (UTF-8 encoded by the caller) and is
    threaded straight into :func:`_anonymized_entity_id` -- see that function's docstring for
    why an unkeyed hash here would not actually anonymize anything.
    """
    adjacency = snapshot.graph.adj
    node_attributes = snapshot.graph.nodes
    result: dict[str, tuple[tuple[RingGraphNode, ...], tuple[RingGraphEdge, ...]]] = {}

    for index, members in enumerate(communities):
        member_set = set(members)
        nodes: dict[str, RingGraphNode] = {
            account: RingGraphNode(node_id=account, kind="account") for account in members
        }
        edges: list[RingGraphEdge] = []
        seen_pairs: set[tuple[str, str]] = set()

        for account in members:
            for neighbour, _data in adjacency[account].items():
                neighbour = str(neighbour)
                is_entity = node_attributes[neighbour].get("kind", "account") == "entity"
                if is_entity:
                    anon = _anonymized_entity_id(neighbour, key)
                    nodes.setdefault(
                        anon,
                        RingGraphNode(
                            node_id=anon,
                            kind="entity",
                            entity_type=str(node_attributes[neighbour].get("spec", "")),
                        ),
                    )
                    pair = (account, anon)
                elif neighbour in member_set:
                    pair = (account, neighbour) if account < neighbour else (neighbour, account)
                else:
                    continue
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append(RingGraphEdge(source=pair[0], target=pair[1]))

        result[f"r{index}"] = (tuple(nodes.values()), tuple(edges))
    return result


class Tier3Result(BaseModel):
    """One Tier-3 decision, as returned to the caller and recorded by the audit trail.

    ``ring_risk_score`` is nullable and that is the point. :class:`app.core.audit.AuditRecord`
    types the field as ``float | None`` for exactly this case: an account with no links is not
    a clean account, it is one this layer cannot speak about.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ring_risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability the account sits in a fraud-bearing ring, from graph "
        "topology alone. Comparable only within one model_version. None when abstained.",
    )
    is_ring_member: bool = Field(
        description="Score at or above the operating threshold chosen on validation by cost. "
        "Always False when abstained — an abstention is not a clearance.",
    )
    is_scoreable: bool = Field(description="Whether the account was in a scoreable ring.")
    ring_id: str | None = Field(default=None, description="None when abstained.")
    ring_size: int = Field(ge=0, description="Member accounts. 0 when abstained.")
    abstention_reason: str | None = Field(default=None)
    snapshot_end: datetime | None = Field(
        default=None,
        description="End of the snapshot this score came from. The gap to the transaction's "
        "own event_time is the score's staleness, which is bounded by the refresh cadence "
        "and reported rather than hidden.",
    )
    latency_ms: float = Field(ge=0.0)
    model_version: str = Field(min_length=1, description="Registry model_id.")


def _betweenness(subgraph: "nx.Graph[str]") -> tuple[dict[str, float], bool]:
    """Return betweenness for one community and whether it was computed exactly."""
    size = subgraph.number_of_nodes()
    if size <= 2:
        return dict.fromkeys(subgraph.nodes(), 0.0), True
    if size <= BETWEENNESS_EXACT_MAX:
        return nx.betweenness_centrality(subgraph, normalized=True), True
    return (
        nx.betweenness_centrality(
            subgraph, k=min(BETWEENNESS_PIVOTS, size), normalized=True, seed=RANDOM_SEED
        ),
        False,
    )


def _accounts_in(graph: "nx.Graph[str]", nodes: Iterable[str]) -> list[str]:
    """Return the account-kind nodes among ``nodes``, in a stable order.

    Sorted, not set-ordered. ``nx.connected_components`` and Louvain both yield *sets*, and
    CPython randomises string hashing per process, so iterating one directly gives a different
    order on every run. Louvain's ``seed`` does not repair that: it seeds the algorithm's own
    randomness, not the order the nodes arrive in. Two runs of the same code with the same seed
    produced ring PR-AUC 0.986566 and 0.986451, and an ``entity_cap=50`` validation score that
    moved by 0.010 -- wider than several of the selection margins it was deciding.
    """
    return sorted(
        str(node) for node in nodes if graph.nodes[node].get("kind", "account") == "account"
    )


def _is_star(subgraph: "nx.Graph[str]") -> bool:
    """Return whether a component has fewer than :data:`MIN_BRANCH_NODES` junction nodes."""
    junctions = sum(1 for _, degree in subgraph.degree() if int(degree) >= 2)
    return junctions < MIN_BRANCH_NODES


def detect_communities(snapshot: GraphSnapshot) -> list[list[str]]:
    """Partition the snapshot into candidate rings.

    Three rules, each of which exists because the alternative was measured and failed:

    * **Per component, not globally.** On PaySim's star forest a global Louvain run spends its
      time rediscovering 353,807 disjoint pieces. Per-component it is linear in the pieces
      that could hold a ring at all.
    * **Stars are not rings** (:data:`MIN_BRANCH_NODES`), on a unipartite graph only. A hub
      with leaves is one popular destination account. Admitting stars would make the detector
      a destination-popularity ranker with a ring-level metric to match. The filter is skipped
      when ``snapshot.is_bipartite``, where the hub is an entity rather than an account and
      several accounts on one device is the structure being looked for; there the entity
      degree cap plays the same role at edge-construction time.
    * **Small components are not sub-partitioned** (:data:`SMALL_COMPONENT_MAX`). Louvain
      shreds short paths; on a 24-step window it destroyed 121 of 121 chain-linked components.
    """
    communities: list[list[str]] = []
    graph = snapshot.graph
    # Components in a stable order, and each one's nodes in a stable order, for the reason
    # given in :func:`_accounts_in`. Without both, ring ids and Louvain's partition of larger
    # components drift between processes.
    components = sorted(
        (sorted(str(node) for node in component) for component in nx.connected_components(graph)),
        key=lambda nodes: (len(nodes), nodes[0] if nodes else ""),
    )
    for component in components:
        accounts = _accounts_in(graph, component)
        if len(accounts) < MIN_RING_SIZE:
            continue
        subgraph = graph.subgraph(component)
        if not snapshot.is_bipartite and _is_star(subgraph):
            continue
        if len(accounts) <= SMALL_COMPONENT_MAX:
            communities.append(sorted(accounts))
            continue
        for community in sorted(
            nx.community.louvain_communities(subgraph, weight="weight", seed=RANDOM_SEED),
            key=lambda nodes: sorted(str(node) for node in nodes)[:1],
        ):
            members = _accounts_in(graph, community)
            if len(members) >= MIN_RING_SIZE:
                communities.append(members)
    return sorted(communities, key=lambda members: (len(members), members[0]))


def ring_feature_frame(
    snapshot: GraphSnapshot, communities: Sequence[Sequence[str]]
) -> pd.DataFrame:
    """Build the per-account structural feature matrix the scorer reads.

    One row per (account, ring). Every column is topology or entity rarity; none reads an
    amount or a balance, which :data:`FORBIDDEN_FEATURE_SOURCES` and its test pin down.
    """
    names = feature_names_for(snapshot.source_dataset)
    rows: list[dict[str, Any]] = []
    # Walk the parent graph's adjacency dict directly. ``Graph.subgraph`` returns a filtered
    # *view*, and every degree, neighbour and edge access through one re-runs the filter; on a
    # 30-day IEEE-CIS window that turned feature extraction into the single slowest stage of
    # the phase. The adjacency mapping is a plain dict of dicts and needs no such machinery.
    adjacency = snapshot.graph.adj
    node_attributes = snapshot.graph.nodes

    for index, members in enumerate(communities):
        member_set = set(members)
        degrees: dict[str, int] = {}
        chain_degrees: dict[str, int] = {}
        entity_degrees: dict[str, int] = {}
        entity_nodes: set[str] = set()
        chain_edges: list[dict[str, Any]] = []
        idf: list[float] = []
        local_edges: list[tuple[str, str]] = []
        internal_count = 0

        for account in members:
            degree = chain_degree = entity_degree = 0
            for neighbour, data in adjacency[account].items():
                neighbour = str(neighbour)
                if node_attributes[neighbour].get("kind", "account") == "entity":
                    entity_nodes.add(neighbour)
                    entity_degree += 1
                    degree += 1
                    idf.append(float(data.get("weight", 0.0)))
                    local_edges.append((account, neighbour))
                elif neighbour in member_set:
                    degree += 1
                    internal_count += 1
                    local_edges.append((account, neighbour))
                    if data.get("chain"):
                        chain_degree += 1
                        chain_edges.append(data)
            degrees[account] = degree
            chain_degrees[account] = chain_degree
            entity_degrees[account] = entity_degree

        # Each account-to-account edge is seen from both ends; entity edges only from one.
        internal = internal_count // 2
        chain_count = len(chain_edges) // 2

        local: nx.Graph[str] = nx.Graph()
        local.add_nodes_from(members)
        local.add_edges_from(local_edges)
        between, exact = _betweenness(local)

        size = len(members)
        specs = {str(node_attributes[node].get("spec", "")) for node in entity_nodes}
        # Density has to mean different things on the two graphs. On PaySim, accounts are
        # adjacent to each other and the ordinary graph density applies. On the bipartite
        # IEEE-CIS graph no two accounts are ever adjacent, so that formula is identically
        # zero and the feature would be dead weight in the scorer; the bipartite fill rate --
        # incidences over accounts times entities -- is the analogue that carries the same
        # meaning of "how completely is this community interconnected".
        if snapshot.is_bipartite:
            possible = float(size * len(entity_nodes))
            realised = float(sum(entity_degrees.values()))
        else:
            possible = float(size * (size - 1) / 2)
            realised = float(internal)

        shared: dict[str, Any] = {
            "ring_size": float(size),
            "ring_density": float(realised / possible) if possible else 0.0,
            "ring_mean_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
            "ring_max_degree": float(max(degrees.values())) if degrees else 0.0,
            "ring_chain_edge_count": float(chain_count),
            "ring_chain_edge_share": (float(chain_count / internal) if internal else 0.0),
            "ring_min_candidate_count": (
                float(min(int(data.get("candidate_count", 0)) for data in chain_edges))
                if chain_edges
                else 0.0
            ),
            "ring_name_corroborated": float(
                any(bool(data.get("name_corroborated", False)) for data in chain_edges)
            ),
            "ring_mean_idf": float(np.mean(idf)) if idf else 0.0,
            "ring_spec_count": float(len({spec for spec in specs if spec})),
            "ring_entity_count": float(len(entity_nodes)),
        }

        for account in members:
            record = dict(shared)
            record.update(
                {
                    "ring_id": f"r{index}",
                    "account_id": account,
                    "account_degree": float(degrees[account]),
                    "account_degree_centrality": (
                        float(degrees[account] / (size - 1)) if size > 1 else 0.0
                    ),
                    "account_betweenness": float(between.get(account, 0.0)),
                    "account_chain_degree": float(chain_degrees[account]),
                    "account_entity_count": float(entity_degrees[account]),
                    "betweenness_exact": exact,
                }
            )
            rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["ring_id", "account_id", "betweenness_exact", *names])
    frame = pd.DataFrame(rows)
    return frame[["ring_id", "account_id", "betweenness_exact", *names]]


@dataclass
class RingScorer:
    """The fitted structural scorer: standardised features into a logistic regression.

    Logistic regression rather than something stronger on purpose. The feature vector is a
    dozen topology measures, the training rows are communities rather than transactions, and
    the question being asked of it — does topology carry signal beyond the edge rule — is
    better served by a model whose coefficients can be read than by one that could memorise
    the graph. It also yields a bounded [0, 1] score, which is what ``AuditRecord`` and the
    Phase 5 meta-learner both expect.
    """

    source_dataset: SourceDataset
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: tuple[str, ...]

    def score_frame(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return P(fraud-bearing ring) per row of a feature frame."""
        if features.empty:
            return np.zeros(0, dtype=np.float64)
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise KeyError(f"feature frame is missing {sorted(missing)}")
        matrix = features[list(self.feature_names)].to_numpy(dtype=np.float64)
        scaled = self.scaler.transform(matrix)
        return np.asarray(self.model.predict_proba(scaled)[:, 1], dtype=np.float64)


def fit_ring_scorer(
    features: pd.DataFrame,
    labels: npt.NDArray[np.bool_],
    source: SourceDataset,
    *,
    seed: int = RANDOM_SEED,
) -> RingScorer:
    """Fit the structural scorer on training-window rings only.

    Args:
        features: Output of :func:`ring_feature_frame` over train snapshots.
        labels: Whether each row's account participated in fraud inside that window.
        source: Which corpus, selecting the feature set.
        seed: Logged with the registry entry.

    Raises:
        ValueError: If the labels carry only one class, which would produce a constant scorer.
    """
    names = feature_names_for(source)
    if features.empty:
        raise ValueError("cannot fit a ring scorer on an empty feature frame")
    if labels.size != len(features):
        raise ValueError(f"labels ({labels.size}) do not align with features ({len(features)})")
    if np.unique(labels).size < 2:
        raise ValueError(
            "training rings carry a single class; the scorer would be constant. Widen the "
            "training window or lower MIN_RING_SIZE."
        )
    matrix = features[list(names)].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs"
    )
    model.fit(scaler.transform(matrix), labels)
    return RingScorer(source_dataset=source, scaler=scaler, model=model, feature_names=names)


def flag_rings(
    graph_snapshot: GraphSnapshot,
    *,
    scorer: RingScorer | None = None,
    threshold: float = 1.0,
) -> list[RingFlag]:
    """Detect communities in a snapshot and describe each as a :class:`RingFlag`.

    The Phase 4 brief's required entry point. ``scorer`` is optional so the same function
    produces the unscored structural output used to build training data, and the scored output
    used at evaluation and serving time.

    Args:
        graph_snapshot: The window to analyse.
        scorer: Fitted structural scorer. When None, ``ring_risk_score`` is None throughout.
        threshold: Operating threshold, chosen on validation by cost. Only used to populate
            ``drivers``; every detected ring is returned either way, so a caller can see what
            was considered and not only what fired.
    """
    communities = detect_communities(graph_snapshot)
    features = ring_feature_frame(graph_snapshot, communities)
    if features.empty:
        return []

    scores = (
        scorer.score_frame(features)
        if scorer is not None
        else np.full(len(features), np.nan, dtype=np.float64)
    )
    features = features.assign(_score=scores)

    def column(group: pd.DataFrame, name: str) -> float:
        """Return a ring-level column, or zero where this corpus does not carry it.

        The two feature sets deliberately differ -- PaySim has chain edges and no entities,
        IEEE-CIS the reverse -- so a flag reads whichever of the two exists rather than
        assuming a union that neither corpus produces.
        """
        return float(group[name].iloc[0]) if name in group.columns else 0.0

    flags: list[RingFlag] = []
    for ring_id, group in features.groupby("ring_id", sort=True):
        best = float(np.nanmax(group["_score"].to_numpy(dtype=np.float64))) if scorer else None
        drivers: list[str] = []
        if best is not None and best >= threshold:
            if column(group, "ring_chain_edge_count") > 0:
                drivers.append("chain_edges")
            if float(group["account_betweenness"].max()) > 0:
                drivers.append("betweenness")
            if float(group["account_degree_centrality"].max()) >= 0.5:
                drivers.append("degree_centrality")
            if column(group, "ring_density") >= 0.5:
                drivers.append("density")
        flags.append(
            RingFlag(
                ring_id=str(ring_id),
                member_account_ids=tuple(str(value) for value in group["account_id"]),
                member_scores=(
                    tuple(float(value) for value in group["_score"]) if scorer is not None else ()
                ),
                size=int(len(group)),
                ring_risk_score=best,
                density=column(group, "ring_density"),
                max_degree_centrality=float(group["account_degree_centrality"].max()),
                max_betweenness=float(group["account_betweenness"].max()),
                betweenness_exact=bool(group["betweenness_exact"].all()),
                chain_edge_count=int(column(group, "ring_chain_edge_count")),
                entity_count=int(column(group, "ring_entity_count")),
                snapshot_end=graph_snapshot.snapshot_end,
                drivers=tuple(drivers),
            )
        )
    return flags


@dataclass
class Tier3Model:
    """The served Tier-3 layer: a precomputed score table plus the threshold that reads it.

    **No graph algorithm runs in the request path.** :meth:`score` is a dictionary lookup, and
    that is a design requirement rather than an optimisation: Phase 7 makes Tier-3 the
    graceful-degradation point with a hard timeout, and a layer that ran Louvain per request
    could not meet any timeout worth setting. Community detection happens in the snapshot job;
    what serving sees is its output.
    """

    model_id: str
    source_dataset: SourceDataset
    threshold: float
    scores: dict[str, float]
    ring_of: dict[str, str]
    ring_sizes: dict[str, int]
    snapshot_end: datetime
    #: Every account the snapshot held, scoreable or not. Without it the two abstention
    #: reasons cannot be told apart -- ``ring_of`` only ever holds scoreable accounts, so an
    #: account seen in a too-small component was reported as one the layer had never seen.
    seen_accounts: frozenset[str] = frozenset()
    scorer: RingScorer | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Per-ring topology for the Phase 8 network view, keyed by the same ring id as
    #: ``ring_of``/``ring_sizes``. Populated by :func:`export_ring_edges` at training time --
    #: see that function's docstring for why entity nodes are hashed here and never reconstructed
    #: at serving time, and ``train_tier3.py``'s ``build_served_model`` for why this is the
    #: *latest* snapshot's topology specifically. Empty for artifacts trained before Phase 8,
    #: which ``load`` tolerates rather than requires re-training every registered model.
    ring_nodes: dict[str, tuple[RingGraphNode, ...]] = field(default_factory=dict)
    ring_edges: dict[str, tuple[RingGraphEdge, ...]] = field(default_factory=dict)

    def score(self, transaction: TransactionFeatures) -> Tier3Result:
        """Return the ring risk for one transaction's account, or an abstention.

        Raises:
            ValueError: If the transaction is from a different corpus. A PaySim ring score
                attached to an IEEE-CIS transaction is the exact cross-corpus leak
                :mod:`app.data.schema` forbids, so it fails loudly rather than returning a
                number that would look plausible in an audit row.
        """
        started = time.perf_counter()
        if transaction.source_dataset != self.source_dataset:
            raise ValueError(
                f"{self.model_id} is fitted on {self.source_dataset} and cannot score a "
                f"{transaction.source_dataset} transaction; ring scores never cross corpora"
            )

        account = transaction.account_id
        if account not in self.scores:
            reason = (
                ABSTAIN_BELOW_MIN_RING if account in self.seen_accounts else ABSTAIN_NOT_IN_SNAPSHOT
            )
            return Tier3Result(
                ring_risk_score=None,
                is_ring_member=False,
                is_scoreable=False,
                ring_id=None,
                ring_size=0,
                abstention_reason=reason,
                snapshot_end=self.snapshot_end,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model_version=self.model_id,
            )

        value = float(self.scores[account])
        ring_id = self.ring_of.get(account)
        return Tier3Result(
            ring_risk_score=value,
            is_ring_member=value >= self.threshold,
            is_scoreable=True,
            ring_id=ring_id,
            ring_size=int(self.ring_sizes.get(ring_id or "", 0)),
            abstention_reason=None,
            snapshot_end=self.snapshot_end,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_version=self.model_id,
        )

    def save(self, directory: Path) -> Path:
        """Write the score table and threshold as JSON.

        JSON rather than a pickle deliberately, matching Tier-2's artefact rule: loading a
        pickle executes arbitrary code, and this path becomes reachable from a Phase 7
        endpoint. A score table is plain data and has no reason to carry executable state.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = artifact_path(self.model_id, directory, ".json")
        payload = {
            "model_id": self.model_id,
            "source_dataset": self.source_dataset,
            "threshold": self.threshold,
            "snapshot_end": self.snapshot_end.isoformat(),
            "scores": self.scores,
            "ring_of": self.ring_of,
            "ring_sizes": self.ring_sizes,
            "seen_accounts": sorted(self.seen_accounts),
            "parameters": self.parameters,
            "ring_nodes": {
                ring_id: [node.model_dump(mode="json") for node in nodes]
                for ring_id, nodes in self.ring_nodes.items()
            },
            "ring_edges": {
                ring_id: [edge.model_dump(mode="json") for edge in edges]
                for ring_id, edges in self.ring_edges.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, model_id: str, directory: Path) -> "Tier3Model":
        """Read back a saved score table.

        Raises:
            ValueError: If ``model_id`` is not a bare registry identifier.
        """
        payload = json.loads(
            artifact_path(model_id, directory, ".json").read_text(encoding="utf-8")
        )
        return cls(
            model_id=str(payload["model_id"]),
            source_dataset=cast(SourceDataset, payload["source_dataset"]),
            threshold=float(payload["threshold"]),
            scores={str(k): float(v) for k, v in payload["scores"].items()},
            ring_of={str(k): str(v) for k, v in payload["ring_of"].items()},
            ring_sizes={str(k): int(v) for k, v in payload["ring_sizes"].items()},
            snapshot_end=datetime.fromisoformat(str(payload["snapshot_end"])),
            seen_accounts=frozenset(str(a) for a in payload.get("seen_accounts", [])),
            parameters=dict(payload.get("parameters", {})),
            ring_nodes={
                str(ring_id): tuple(RingGraphNode.model_validate(node) for node in nodes)
                for ring_id, nodes in payload.get("ring_nodes", {}).items()
            },
            ring_edges={
                str(ring_id): tuple(RingGraphEdge.model_validate(edge) for edge in edges)
                for ring_id, edges in payload.get("ring_edges", {}).items()
            },
        )


def build_score_table(
    flags: Iterable[RingFlag],
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    """Flatten ring flags into the per-account lookup tables :class:`Tier3Model` serves.

    **The one place the per-account rule lives.** An account is scored by *its own* highest
    score across the rings it belongs to -- not by the highest score any member of its ring
    reached. The distinction is not cosmetic: :class:`RingFlag.ring_risk_score` is a ring-level
    summary used for ring-level evaluation, and assigning it to every member hands a quiet
    account the score of the loudest one in its component. An earlier version did exactly that
    on the serving path while the offline metric took the per-account maximum, so the number
    this layer reported and the number it served were different quantities.

    ``member_scores`` is aligned to ``member_account_ids``; a flag without it falls back to the
    ring score, which is the pre-scoring structural path where no per-member score exists.
    """
    scores: dict[str, float] = {}
    ring_of: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for flag in flags:
        sizes[flag.ring_id] = flag.size
        if flag.ring_risk_score is None:
            continue
        members = flag.member_account_ids
        per_member = (
            flag.member_scores
            if len(flag.member_scores) == len(members)
            else (float(flag.ring_risk_score),) * len(members)
        )
        for account, value in zip(members, per_member, strict=True):
            if account not in scores or float(value) > scores[account]:
                scores[account] = float(value)
                ring_of[account] = flag.ring_id
    return scores, ring_of, sizes


def benchmark_latency(
    model: Tier3Model,
    transactions: Sequence[TransactionFeatures],
    *,
    calls: int = LATENCY_BENCHMARK_CALLS,
) -> dict[str, float]:
    """Measure per-call scoring latency, matching Tier-1's and Tier-2's benchmark shape."""
    if not transactions:
        raise ValueError("latency benchmark needs at least one transaction")
    # Read the latency the result reports rather than timing the call from outside, so this
    # figure is measured the same way Tier-2's is and the two are comparable.
    timings: list[float] = []
    for index in range(calls):
        result = model.score(transactions[index % len(transactions)])
        timings.append(result.latency_ms)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "calls": float(calls),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "max_ms": float(values.max()),
        "budget_p95_ms": LATENCY_BUDGET_P95_MS,
    }
