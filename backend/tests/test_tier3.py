"""Tier-3 graph layer tests.

Every leakage guard here is paired with a guard-on-the-guard that plants a violation and
asserts the check fires. A test that walks a graph looking for a future read passes trivially
if the comparison it makes is broken, so each one is followed by proof that it can fail.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from app.data.schema import TransactionFeatures
from app.ml.cost import CostModel, cost_at_threshold
from app.ml.registry import artifact_path
from app.models import train_tier3 as driver
from app.models.tier3_edges import (
    CASH_OUT,
    DEFAULT_MAX_ENTITY_DEGREE,
    IEEE_FINGERPRINTS,
    NON_CIRCULAR_FINGERPRINTS,
    TRANSFER,
    UID_COLUMNS,
    FingerprintSpec,
    build_chain_edges,
    build_entity_edges,
    fingerprint_keys,
)
from app.models.tier3_graph import (
    ABSTAIN_BELOW_MIN_RING,
    ABSTAIN_NOT_IN_SNAPSHOT,
    FORBIDDEN_FEATURE_SOURCES,
    MIN_RING_SIZE,
    SMALL_COMPONENT_MAX,
    GraphSnapshot,
    IEEECISSharedEntityGraph,
    PaySimMoneyFlowGraph,
    RingScorer,
    Tier3Model,
    benchmark_latency,
    build_score_table,
    detect_communities,
    export_ring_edges,
    feature_names_for,
    fit_ring_scorer,
    flag_rings,
    ring_feature_frame,
)

EPOCH = datetime(2017, 1, 1, tzinfo=UTC)

#: Not a secret: it signs nothing that exists outside the test process. See
#: tests/conftest.py's TEST_SIGNING_KEY for the same reasoning applied to the JWT key.
TEST_ENTITY_ANONYMIZATION_KEY = b"test-only-entity-anonymization-key-not-a-real-secret"

#: `ieee_rows` defaults every fingerprint spec's columns to the same constant across rows, so a
#: fixture wanting exactly one shared entity (rather than one per matching spec: card_fp,
#: device_fp, email_dist all firing at once) has to narrow the graph to a single spec.
DEVICE_ONLY_SPEC = tuple(spec for spec in IEEE_FINGERPRINTS if spec.name == "device_fp")


def paysim_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a canonical PaySim frame from compact row specifications."""
    frame = pd.DataFrame(rows)
    frame["event_time"] = [EPOCH + timedelta(hours=int(step) - 1) for step in frame["step"]]
    frame["transaction_id"] = [f"t{index}" for index in range(len(frame))]
    frame["source_dataset"] = "paysim"
    frame["feature_version"] = "fv_test"
    # Rows that omit the key arrive as NaN rather than False, which would make is_fraud an
    # object column and break every boolean mask downstream.
    frame["is_fraud"] = (
        frame["is_fraud"].fillna(False).astype(bool) if "is_fraud" in frame.columns else False
    )
    if "split" not in frame.columns:
        frame["split"] = "train"
    return frame


def chain_frame() -> pd.DataFrame:
    """One transfer/cash-out chain plus one pure star, with nothing else in the window.

    The star exists so the star filter has something to reject; the chain exists so it has
    something to keep. Both cases are asserted present by the fixture self-check below.
    """
    return paysim_rows(
        [
            # The chain: victim -> muleA, then muleB -> destination for the same amount.
            {
                "account_id": "victim",
                "counterparty_id": "muleA",
                "transaction_type": TRANSFER,
                "amount": 100.0,
                "step": 5,
                "is_fraud": True,
            },
            {
                "account_id": "muleB",
                "counterparty_id": "cashdest",
                "transaction_type": CASH_OUT,
                "amount": 100.0,
                "step": 5,
                "is_fraud": True,
            },
            # A pure star: three unrelated origins paying one popular destination.
            {
                "account_id": "s1",
                "counterparty_id": "hub",
                "transaction_type": CASH_OUT,
                "amount": 11.0,
                "step": 5,
            },
            {
                "account_id": "s2",
                "counterparty_id": "hub",
                "transaction_type": CASH_OUT,
                "amount": 12.0,
                "step": 5,
            },
            {
                "account_id": "s3",
                "counterparty_id": "hub",
                "transaction_type": CASH_OUT,
                "amount": 13.0,
                "step": 5,
            },
        ]
    )


def ieee_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a canonical IEEE-CIS frame from compact row specifications."""
    frame = pd.DataFrame(rows)
    frame["event_time"] = [EPOCH + timedelta(hours=index) for index in range(len(frame))]
    frame["transaction_id"] = [f"i{index}" for index in range(len(frame))]
    frame["source_dataset"] = "ieee_cis"
    frame["feature_version"] = "fv_test"
    frame["amount"] = 10.0
    frame["is_fraud"] = (
        frame["is_fraud"].fillna(False).astype(bool) if "is_fraud" in frame.columns else False
    )
    if "split" not in frame.columns:
        frame["split"] = "train"
    for column in ("card1", "card2", "card5", "addr1", "dist1"):
        if column not in frame.columns:
            frame[column] = 1.0
    for column in ("DeviceInfo", "id_30", "id_31", "id_33", "P_emaildomain"):
        if column not in frame.columns:
            frame[column] = "x"
    return frame


def snapshot_of(frame: pd.DataFrame, *, step_window: int = 1) -> object:
    """Build a PaySim snapshot ending just after the frame's last event."""
    graph = PaySimMoneyFlowGraph(step_window=step_window, amount_tolerance=0.0)
    graph.insert(frame)
    end = frame["event_time"].max() + timedelta(seconds=1)
    return graph.snapshot(end.to_pydatetime())


# --- Time discipline -------------------------------------------------------------------


def test_snapshot_refuses_a_transaction_at_or_after_its_own_end() -> None:
    """A snapshot may only be built from strictly earlier transactions."""
    frame = chain_frame()
    graph = PaySimMoneyFlowGraph(step_window=1, amount_tolerance=0.0)
    graph.insert(frame)
    end = frame["event_time"].max().to_pydatetime()
    with pytest.raises(ValueError, match="strictly earlier"):
        graph.snapshot(end)


def test_future_read_guard_actually_catches_one() -> None:
    """Guard on the guard: plant a future row and confirm the check fires.

    Without this, :func:`test_snapshot_refuses_a_transaction_at_or_after_its_own_end` would
    still pass if the comparison were inverted or dropped, because a correct snapshot also
    raises nothing when the buffer is empty.
    """
    frame = chain_frame()
    end = (frame["event_time"].max() + timedelta(hours=1)).to_pydatetime()

    clean = PaySimMoneyFlowGraph(step_window=1, amount_tolerance=0.0)
    clean.insert(frame)
    clean.snapshot(end)  # the honest case must not raise

    planted = frame.copy()
    planted.loc[planted.index[0], "event_time"] = pd.Timestamp(end) + pd.Timedelta(hours=1)
    dirty = PaySimMoneyFlowGraph(step_window=1, amount_tolerance=0.0)
    dirty.insert(planted)
    with pytest.raises(ValueError):
        dirty.snapshot(end)


def test_chain_edge_never_links_a_cashout_earlier_than_its_transfer() -> None:
    """A cash-out preceding its transfer is not a drain and must not become an edge."""
    frame = paysim_rows(
        [
            {
                "account_id": "early_out",
                "counterparty_id": "d",
                "transaction_type": CASH_OUT,
                "amount": 100.0,
                "step": 1,
            },
            {
                "account_id": "victim",
                "counterparty_id": "muleA",
                "transaction_type": TRANSFER,
                "amount": 100.0,
                "step": 9,
            },
        ]
    )
    edges = build_chain_edges(frame, step_window=3, amount_tolerance=0.0)
    assert edges.frame.empty, "a cash-out eight steps before the transfer must not match"

    # The same amount placed *after* the transfer does match, proving the window works at all
    # rather than the matcher simply returning nothing.
    later = frame.copy()
    later.loc[later.index[0], "step"] = 10
    later.loc[later.index[0], "event_time"] = EPOCH + timedelta(hours=9)
    matched = build_chain_edges(later, step_window=3, amount_tolerance=0.0)
    assert not matched.frame.empty
    assert (matched.frame["step_gap"] >= 0).all()


def test_evict_before_removes_only_expired_edges() -> None:
    """Eviction drops what left the window and keeps what did not."""
    frame = chain_frame()
    graph = PaySimMoneyFlowGraph(step_window=1, amount_tolerance=0.0)
    graph.insert(frame)
    cutoff = frame["event_time"].min() + timedelta(hours=1)
    evicted = graph.evict_before(cutoff.to_pydatetime())
    assert evicted == int((frame["event_time"] < cutoff).sum())
    assert (graph.buffered()["event_time"] >= cutoff).all()


def corpus_frame() -> pd.DataFrame:
    """A small PaySim-shaped corpus with two structurally distinct ring types per window.

    Each hour carries a fraudulent ring and a legitimate amount collision, and the two differ
    in *shape*: the fraud ring funnels two transfers into one mule before the cash-out, so it
    is larger and carries a higher-degree junction, while the coincidence is a plain four-node
    chain. That difference is the only thing a structural scorer could key on, which is the
    point -- an earlier version of this fixture made both types identical four-node chains and
    every PR-AUC came out at exactly the base rate, because there was nothing to separate.
    """
    rows: list[dict[str, object]] = []
    for hour in range(36):
        amount = 1000.0 + hour
        rows += [
            # Fraud: two victims funnel into one mule, which is then drained.
            {
                "account_id": f"fv{hour}a",
                "counterparty_id": f"fm{hour}",
                "transaction_type": TRANSFER,
                "amount": amount,
                "step": hour + 1,
                "is_fraud": True,
            },
            {
                "account_id": f"fv{hour}b",
                "counterparty_id": f"fm{hour}",
                "transaction_type": TRANSFER,
                "amount": amount,
                "step": hour + 1,
                "is_fraud": True,
            },
            {
                "account_id": f"fo{hour}",
                "counterparty_id": f"fd{hour}",
                "transaction_type": CASH_OUT,
                "amount": amount,
                "step": hour + 1,
                "is_fraud": True,
            },
            # Legitimate: one transfer that happens to share an amount with one cash-out.
            {
                "account_id": f"lv{hour}",
                "counterparty_id": f"lm{hour}",
                "transaction_type": TRANSFER,
                "amount": 5000.0 + hour,
                "step": hour + 1,
                "is_fraud": False,
            },
            {
                "account_id": f"lo{hour}",
                "counterparty_id": f"ld{hour}",
                "transaction_type": CASH_OUT,
                "amount": 5000.0 + hour,
                "step": hour + 1,
                "is_fraud": False,
            },
        ]
    frame = paysim_rows(rows)
    hours = frame["step"] - 1
    frame["split"] = np.where(hours < 18, "train", np.where(hours < 27, "val", "test"))
    return frame


def run_small_corpus(frame: pd.DataFrame) -> driver.CorpusReport:
    """Run the real selection path over a small corpus."""
    return driver.run_corpus(
        "paysim",
        frame,
        [("sw=0", {"step_window": 0, "amount_tolerance": 0.0}, driver.paysim_factory(0, 0.0))],
        cadence=pd.Timedelta(hours=6),
        window=pd.Timedelta(hours=12),
        cost_model=CostModel(),
    )


@pytest.fixture(scope="module")
def baseline_report() -> driver.CorpusReport:
    """Run the corpus once and share it: each run builds a dozen graphs."""
    return run_small_corpus(corpus_frame())


def test_corpus_fixture_carries_two_classes_and_two_shapes() -> None:
    """Fixture self-validation: the selection tests below are vacuous without both."""
    frame = corpus_frame()
    for split in ("train", "val", "test"):
        subset = frame.loc[frame["split"] == split]
        assert not subset.empty, f"{split} split is empty"
        assert subset["is_fraud"].nunique() == 2, f"{split} split carries a single class"
    funnelled = frame.loc[frame["is_fraud"] & (frame["transaction_type"] == TRANSFER)]
    assert (
        funnelled.groupby("counterparty_id").size().max() == 2
    ), "fraud rings must differ in shape from the coincidences, or nothing can separate them"


def test_threshold_is_chosen_on_validation_not_test(
    baseline_report: driver.CorpusReport,
) -> None:
    """Corrupting the test split must not move the operating threshold.

    The standing convention from ``test_tier1.py`` and ``test_tier2.py``, run here through the
    real selection path rather than against the threshold helper in isolation. Phase 2's log
    records a phase brief read literally enough to select on test; this is the wire that
    catches the same mistake in Phase 4.
    """
    corrupted = corpus_frame()
    test_rows = corrupted["split"] == "test"
    corrupted.loc[test_rows, "is_fraud"] = ~corrupted.loc[test_rows, "is_fraud"].to_numpy(bool)
    moved = run_small_corpus(corrupted)

    assert moved.threshold == pytest.approx(
        baseline_report.threshold
    ), "the operating threshold moved when only the test split changed"
    assert moved.winner.val_ring_pr_auc == pytest.approx(baseline_report.winner.val_ring_pr_auc)


def test_selection_guard_would_catch_a_planted_validation_leak(
    baseline_report: driver.CorpusReport,
) -> None:
    """Guard on the guard: changing the *validation* split must change the selection.

    Without this, the test above would still pass if the threshold were a hard-coded constant
    that read neither split.
    """
    shifted = corpus_frame()
    validation = shifted["split"] == "val"
    shifted.loc[validation, "is_fraud"] = ~shifted.loc[validation, "is_fraud"].to_numpy(bool)
    moved = run_small_corpus(shifted)

    assert moved.winner.val_ring_pr_auc != pytest.approx(
        baseline_report.winner.val_ring_pr_auc
    ), "the selection ignored the validation split entirely"


def test_paysim_is_evaluated_per_ring_because_its_origins_are_near_unique(
    baseline_report: driver.CorpusReport,
) -> None:
    """The unit of analysis is chosen on measured coverage, not declared in advance.

    PaySim origins have degree 1 in 99.95% of cases, so an account seen in one snapshot window
    essentially never returns in the next and a per-transaction ring score abstains on
    everything. The corpus report must say so rather than publishing a transaction metric
    computed almost entirely from abstentions.
    """
    assert baseline_report.unit == "ring"
    assert baseline_report.abstention_rate >= driver.TRANSACTION_COVERAGE_FLOOR
    assert (
        baseline_report.transaction_test is None
    ), "a transaction-level headline was published on a corpus that abstains on all of them"
    assert baseline_report.ring_test is not None
    assert any("unit of analysis" in note for note in baseline_report.notes)


def test_ring_evaluation_carries_a_base_rate_and_a_cost_estimate(
    baseline_report: driver.CorpusReport,
) -> None:
    """ml-evaluation-standards: precision without a base rate is uninterpretable, and a
    metrics section without a false-positive cost is incomplete for this project."""
    assert baseline_report.ring_test is not None
    assert 0.0 < baseline_report.ring_test.base_rate < 1.0
    assert baseline_report.ring_test.false_positive_cost is not None
    assert baseline_report.ring_test.split == "test"


# --- What counts as a ring -------------------------------------------------------------


def test_fixture_contains_both_a_chain_and_a_star() -> None:
    """Fixture self-validation: the cases the next two tests rely on must actually be there."""
    frame = chain_frame()
    assert (frame["transaction_type"] == TRANSFER).any(), "fixture must contain a transfer"
    assert (
        frame["counterparty_id"] == "hub"
    ).sum() >= MIN_RING_SIZE, (
        "fixture must contain a star large enough to be rejected on structure, not on size"
    )


def test_a_pure_star_is_not_a_ring() -> None:
    """A popular destination is one account, not a collusion structure."""
    snapshot = snapshot_of(chain_frame())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    members = {account for community in communities for account in community}
    assert "hub" not in members, "a destination star was reported as a ring"


def test_a_chain_linked_component_is_a_ring() -> None:
    """The inferred chain edge is what makes a multi-hop structure detectable."""
    snapshot = snapshot_of(chain_frame())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    members = {account for community in communities for account in community}
    assert {"victim", "muleA", "muleB", "cashdest"} <= members


def test_small_components_are_not_shredded_by_louvain() -> None:
    """Louvain maximises modularity, which is meaningless on a four-node path.

    Measured on a 24-step PaySim window, sub-partitioning destroyed 121 of 121 chain-linked
    components by splitting them below the minimum ring size.
    """
    snapshot = snapshot_of(chain_frame())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    assert communities, "the chain component was partitioned out of existence"
    largest = max(len(community) for community in communities)
    assert largest >= 4
    assert largest <= SMALL_COMPONENT_MAX


def test_name_corroboration_is_recorded_but_never_required() -> None:
    """Phase 1 measured name continuity at 0.00%; requiring it would find nothing."""
    frame = chain_frame()
    edges = build_chain_edges(frame, step_window=1, amount_tolerance=0.0)
    assert not edges.frame.empty, "matching required a name match it should treat as optional"
    assert not bool(edges.frame["name_corroborated"].any())
    assert "name_corroborated" in edges.frame.columns


def test_ambiguous_matches_are_downweighted_not_dropped() -> None:
    """Ambiguity belongs on the edge; dropping hard cases would flatter precision."""
    frame = paysim_rows(
        [
            {
                "account_id": "v",
                "counterparty_id": "m",
                "transaction_type": TRANSFER,
                "amount": 50.0,
                "step": 1,
            },
            *[
                {
                    "account_id": f"o{index}",
                    "counterparty_id": f"d{index}",
                    "transaction_type": CASH_OUT,
                    "amount": 50.0,
                    "step": 1,
                }
                for index in range(4)
            ],
        ]
    )
    edges = build_chain_edges(frame, step_window=0, amount_tolerance=0.0)
    assert len(edges.frame) == 4, "every candidate partner must be emitted"
    assert (edges.frame["candidate_count"] == 4).all()
    assert np.allclose(edges.frame["weight"].to_numpy(), 0.25)


# --- Abstention ------------------------------------------------------------------------


def fitted_model(frame: pd.DataFrame) -> Tier3Model:
    """Fit a scorer on a frame and freeze it into a served model."""
    snapshot = snapshot_of(frame)
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    features = ring_feature_frame(snapshot, communities)  # type: ignore[arg-type]
    fraud = driver.fraud_accounts(frame)
    labels = features["account_id"].isin(fraud).to_numpy(dtype=bool)
    if labels.all() or not labels.any():
        # Give the scorer two classes without touching the topology under test.
        labels = np.array([index % 2 == 0 for index in range(len(features))], dtype=bool)
    scorer = fit_ring_scorer(features, labels, "paysim")
    flags = flag_rings(snapshot, scorer=scorer, threshold=0.5)  # type: ignore[arg-type]
    scores, ring_of, sizes = build_score_table(flags)
    return Tier3Model(
        model_id="tier3-graph-louvain-paysim-test",
        source_dataset="paysim",
        threshold=0.5,
        scores=scores,
        ring_of=ring_of,
        ring_sizes=sizes,
        snapshot_end=snapshot.snapshot_end,  # type: ignore[attr-defined]
        scorer=scorer,
    )


def transaction(account: str, source: str = "paysim") -> TransactionFeatures:
    """Build a minimal transaction for one account."""
    return TransactionFeatures(
        transaction_id="probe",
        source_dataset=source,  # type: ignore[arg-type]
        event_time=EPOCH + timedelta(hours=20),
        amount=Decimal("1.0000"),
        account_id=account,
        counterparty_id=None,
        transaction_type=None,
        feature_version="fv_test",
    )


def test_unseen_accounts_abstain_rather_than_scoring_zero() -> None:
    """An account with no links is one Tier-3 has no opinion on, not a clean one."""
    model = fitted_model(chain_frame())
    result = model.score(transaction("an-account-never-seen"))
    assert result.ring_risk_score is None, "abstention was encoded as a score"
    assert result.is_scoreable is False
    assert result.is_ring_member is False
    assert result.abstention_reason is not None


def test_abstained_accounts_rank_last_and_never_flag() -> None:
    """The ranking sentinel must sit below every real score."""
    scored = pd.DataFrame(
        {
            "is_fraud": [True, False, True],
            "ring_risk_score": [0.4, np.nan, 0.6],
        }
    )
    _, ranked = driver.transaction_scores(scored)
    assert ranked[1] == driver.ABSTAINED_RANK_SENTINEL
    assert ranked[1] < ranked.max()
    assert driver.ABSTAINED_RANK_SENTINEL < 0.0


def test_a_scored_account_is_actually_scored() -> None:
    """Guard on the abstention tests: the model must score somebody, or they prove nothing."""
    frame = chain_frame()
    model = fitted_model(frame)
    assert model.scores, "fixture produced no scoreable accounts"
    account = next(iter(model.scores))
    result = model.score(transaction(account))
    assert result.is_scoreable is True
    assert result.ring_risk_score is not None
    assert 0.0 <= result.ring_risk_score <= 1.0


# --- Cross-corpus and artefact safety ---------------------------------------------------


def test_a_paysim_ring_score_never_attaches_to_an_ieee_cis_transaction() -> None:
    """``app.data.schema`` states this as a rule; here it is enforced rather than described."""
    model = fitted_model(chain_frame())
    with pytest.raises(ValueError, match="never cross corpora"):
        model.score(transaction("victim", source="ieee_cis"))


@pytest.mark.parametrize(
    "model_id",
    ["../escape", "sub/dir", "..", ".", "", "a\\b", "C:secret", "/etc/passwd", "C:/x"],
)
def test_load_refuses_a_path_traversing_model_id(model_id: str, tmp_path: Path) -> None:
    """The artefact path is built from the id, so a traversing id would escape the directory."""
    with pytest.raises(ValueError):
        Tier3Model.load(model_id, tmp_path)


def test_saved_artifact_is_plain_json_not_a_pickle(tmp_path: Path) -> None:
    """Loading a pickle executes arbitrary code, and Phase 7 makes this path reachable."""
    import json

    model = fitted_model(chain_frame())
    path = model.save(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_id"] == model.model_id
    assert set(payload) >= {"scores", "ring_of", "threshold", "source_dataset"}

    restored = Tier3Model.load(model.model_id, tmp_path)
    assert restored.scores == model.scores
    assert restored.threshold == model.threshold


def test_ring_topology_survives_a_save_and_load_round_trip(tmp_path: Path) -> None:
    """Phase 8's network view reads `ring_nodes`/`ring_edges` off a loaded artifact."""
    from app.models.tier3_graph import RingGraphEdge, RingGraphNode

    model = fitted_model(chain_frame())
    ring_id = next(iter(model.ring_of.values()))
    with_topology = replace(
        model,
        ring_nodes={ring_id: (RingGraphNode(node_id="victim", kind="account"),)},
        ring_edges={ring_id: (RingGraphEdge(source="victim", target="muleA"),)},
    )

    with_topology.save(tmp_path)
    restored = Tier3Model.load(with_topology.model_id, tmp_path)

    assert restored.ring_nodes == with_topology.ring_nodes
    assert restored.ring_edges == with_topology.ring_edges


def test_an_artifact_saved_before_phase_8_loads_with_empty_topology(tmp_path: Path) -> None:
    """A registered artifact from before `ring_nodes`/`ring_edges` existed must still load."""
    import json

    model = fitted_model(chain_frame())
    path = model.save(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["ring_nodes"]
    del payload["ring_edges"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = Tier3Model.load(model.model_id, tmp_path)
    assert restored.ring_nodes == {}
    assert restored.ring_edges == {}


# --- The structural-only invariant -------------------------------------------------------


@pytest.mark.parametrize("source", ["paysim", "ieee_cis"])
def test_no_ring_feature_is_named_after_money(source: str) -> None:
    """The scorer reads topology. Amount enters only through which chain edges exist."""
    names = set(feature_names_for(source))  # type: ignore[arg-type]
    assert not (names & FORBIDDEN_FEATURE_SOURCES)


def test_ring_features_are_unchanged_when_every_amount_is_rescaled() -> None:
    """Scaling all amounts preserves exact-match chains, so topology -- and the features
    computed from it -- must be bit-identical.

    A behavioural check rather than a naming one: a feature could read an amount without
    saying so in its name, and this is what would catch it.
    """
    frame = chain_frame()
    snapshot = snapshot_of(frame)
    base = ring_feature_frame(snapshot, detect_communities(snapshot))  # type: ignore[arg-type]

    rescaled = frame.copy()
    rescaled["amount"] = rescaled["amount"] * 1000.0
    other = snapshot_of(rescaled)
    after = ring_feature_frame(other, detect_communities(other))  # type: ignore[arg-type]

    numeric = list(feature_names_for("paysim"))
    pd.testing.assert_frame_equal(
        base[["account_id", *numeric]].sort_values("account_id").reset_index(drop=True),
        after[["account_id", *numeric]].sort_values("account_id").reset_index(drop=True),
    )


def test_rescaling_guard_would_notice_a_planted_amount_feature() -> None:
    """Guard on the guard: an amount-derived column does differ after rescaling."""
    frame = chain_frame()
    snapshot = snapshot_of(frame)
    base = ring_feature_frame(snapshot, detect_communities(snapshot))  # type: ignore[arg-type]
    rescaled = frame.copy()
    rescaled["amount"] = rescaled["amount"] * 1000.0
    other = snapshot_of(rescaled)
    after = ring_feature_frame(other, detect_communities(other))  # type: ignore[arg-type]

    planted_base = base.assign(planted=float(frame["amount"].sum()))
    planted_after = after.assign(planted=float(rescaled["amount"].sum()))
    assert planted_base["planted"].iloc[0] != planted_after["planted"].iloc[0]


# --- IEEE-CIS entity graph ---------------------------------------------------------------


def test_hub_entities_above_the_cap_create_no_edges() -> None:
    """Without the cap, ``card4``'s four values put 98,466 accounts in one component."""
    frame = ieee_rows([{"account_id": f"a{index}"} for index in range(12)])
    capped = build_entity_edges(frame, IEEE_FINGERPRINTS, max_entity_degree=5)
    assert capped.frame.empty, "an entity linking 12 accounts survived a cap of 5"

    uncapped = build_entity_edges(frame, IEEE_FINGERPRINTS, max_entity_degree=50)
    assert not uncapped.frame.empty, "the fixture's entity never linked anything at all"


def test_entity_degree_cap_rejects_a_planted_hub_but_keeps_a_small_entity() -> None:
    """Guard on the guard: the cap must be selective, not simply destructive."""
    rows: list[dict[str, object]] = [{"account_id": f"hub{index}"} for index in range(30)]
    rows += [
        {"account_id": f"pair{index}", "DeviceInfo": "rare", "id_31": "rare"} for index in range(2)
    ]
    frame = ieee_rows(rows)
    edges = build_entity_edges(frame, IEEE_FINGERPRINTS, max_entity_degree=10)
    assert not edges.frame.empty, "the small entity was destroyed along with the hub"
    linked = set(edges.frame["account_id"])
    assert {"pair0", "pair1"} <= linked
    assert not any(account.startswith("hub") for account in linked)


def test_an_entity_hub_is_a_ring_on_the_bipartite_graph() -> None:
    """The star filter must not apply where the hub is an entity rather than an account.

    On PaySim a star's centre is a destination account and the star is one popular
    destination. Here the centre is a shared device, and several accounts on one device is
    precisely the structure this graph exists to surface.
    """
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(4)
        ]
    )
    graph = IEEECISSharedEntityGraph(max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE)
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)
    assert communities, "an entity-centred community was filtered out as a star"
    assert len(communities[0]) >= MIN_RING_SIZE


def test_ring_size_counts_accounts_not_entity_nodes() -> None:
    """A ring of three accounts joined by two devices is a ring of three."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(3)
        ]
    )
    graph = IEEECISSharedEntityGraph(max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE)
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)
    features = ring_feature_frame(snapshot, communities)
    assert not features.empty
    assert set(features["ring_size"].unique()) == {3.0}
    assert (features["ring_entity_count"] > 0).all()


# --- Ring topology export (Phase 8 network view) ------------------------------------------


def test_export_ring_edges_hashes_entity_ids_and_never_leaks_the_raw_fingerprint() -> None:
    """The Phase 8 spec's non-negotiable: an entity node id is never the raw composite key."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "very-secret-device", "id_31": "shared"}
            for index in range(3)
        ]
    )
    graph = IEEECISSharedEntityGraph(max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE)
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    topology = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]

    assert topology, "no ring topology was exported for a fixture built to contain one"
    nodes, edges = next(iter(topology.values()))
    entity_nodes = [node for node in nodes if node.kind == "entity"]
    account_nodes = [node for node in nodes if node.kind == "account"]

    assert entity_nodes, "no entity node was exported on a bipartite fixture"
    assert {node.node_id for node in account_nodes} == {"a0", "a1", "a2"}
    for node in entity_nodes:
        assert "very-secret-device" not in node.node_id
        assert "shared" not in node.node_id
        assert len(node.node_id) == 16, "the exported id must be the truncated hash, not the key"
        assert node.entity_type, "an entity node must carry which fingerprint spec produced it"

    node_ids = {node.node_id for node in nodes}
    for edge in edges:
        assert edge.source in node_ids
        assert edge.target in node_ids


def test_export_ring_edges_hashing_is_deterministic_and_collision_free_within_a_ring() -> None:
    """The same raw entity must anonymize to the same id everywhere it appears in one export."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "device-x", "id_31": "shared"}
            for index in range(4)
        ]
    )
    graph = IEEECISSharedEntityGraph(
        specs=DEVICE_ONLY_SPEC, max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE
    )
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    topology = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]

    nodes, edges = next(iter(topology.values()))
    entity_ids = {node.node_id for node in nodes if node.kind == "entity"}
    assert len(entity_ids) == 1, "one shared device must anonymize to exactly one node id"
    entity_id = next(iter(entity_ids))
    assert sum(1 for edge in edges if entity_id in (edge.source, edge.target)) == 4


def test_export_ring_edges_ring_ids_match_ring_feature_frames() -> None:
    """The exported ring ids must line up with `ring_feature_frame`'s, both `f"r{index}"`."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(3)
        ]
    )
    graph = IEEECISSharedEntityGraph(max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE)
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    features = ring_feature_frame(snapshot, communities)  # type: ignore[arg-type]
    topology = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]

    assert set(features["ring_id"].unique()) == set(topology)


def test_export_ring_edges_on_a_paysim_chain_has_account_edges_and_no_entities() -> None:
    """PaySim carries chain edges between accounts, never an entity node -- the mirror case."""
    frame = chain_frame()
    snapshot = snapshot_of(frame)
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    topology = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]

    assert topology, "no ring topology was exported for a fixture built to contain a chain"
    for nodes, edges in topology.values():
        assert all(node.kind == "account" for node in nodes)
        assert edges, "a PaySim ring with no edges at all is not a ring"


def test_export_ring_edges_excludes_edges_reaching_outside_the_community() -> None:
    """An entity shared with an account outside this ring belongs to a different ring's picture."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(3)
        ]
        + [{"account_id": "outsider", "DeviceInfo": "unrelated", "id_31": "unrelated"}]
    )
    graph = IEEECISSharedEntityGraph(
        specs=DEVICE_ONLY_SPEC, max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE
    )
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]
    topology = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]

    all_node_ids = {node.node_id for nodes, _edges in topology.values() for node in nodes}
    assert "outsider" not in all_node_ids


def test_entity_ids_change_with_the_key_and_do_not_match_an_unkeyed_hash() -> None:
    """The actual security-review fix: an unkeyed hash over IEEE-CIS's small, public
    fingerprint domain is a dictionary attack away from the raw card/device fingerprint --
    keying it with a deployment secret is what makes the mapping non-reversible.
    """
    import hashlib

    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(3)
        ]
    )
    graph = IEEECISSharedEntityGraph(
        specs=DEVICE_ONLY_SPEC, max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE
    )
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]

    first = export_ring_edges(snapshot, communities, key=b"key-one")  # type: ignore[arg-type]
    second = export_ring_edges(snapshot, communities, key=b"key-two")  # type: ignore[arg-type]

    entity_id_one = next(
        node.node_id for nodes, _edges in first.values() for node in nodes if node.kind == "entity"
    )
    entity_id_two = next(
        node.node_id for nodes, _edges in second.values() for node in nodes if node.kind == "entity"
    )
    assert entity_id_one != entity_id_two, "two different keys must not anonymize to the same id"

    # The raw fingerprint the entity node was built from -- read directly off the graph rather
    # than reconstructed by hand, so this assertion does not depend on independently guessing
    # fingerprint_keys's exact join format. An attacker holding the public IEEE-CIS corpus can
    # build this same candidate exactly as cheaply as this test just did.
    raw_entity = next(
        str(node)
        for node in snapshot.graph.nodes  # type: ignore[attr-defined]
        if snapshot.graph.nodes[node].get("kind") == "entity"  # type: ignore[attr-defined]
    )
    unkeyed = hashlib.sha256(raw_entity.encode("utf-8")).hexdigest()[:16]
    assert (
        entity_id_one != unkeyed
    ), "the keyed id must not degrade to the plain SHA-256 anyone can compute"


def test_entity_ids_are_stable_for_the_same_key_and_entity() -> None:
    """Same key, same raw entity, same node id -- required so one ring's nodes can be joined
    against another export from the same deployment (the same key), not just internally
    consistent within a single export_ring_edges call."""
    frame = ieee_rows(
        [
            {"account_id": f"a{index}", "DeviceInfo": "shared", "id_31": "shared"}
            for index in range(3)
        ]
    )
    graph = IEEECISSharedEntityGraph(
        specs=DEVICE_ONLY_SPEC, max_entity_degree=DEFAULT_MAX_ENTITY_DEGREE
    )
    graph.insert(frame)
    snapshot = graph.snapshot((frame["event_time"].max() + timedelta(seconds=1)).to_pydatetime())
    communities = detect_communities(snapshot)  # type: ignore[arg-type]

    first = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]
    second = export_ring_edges(snapshot, communities, key=TEST_ENTITY_ANONYMIZATION_KEY)  # type: ignore[arg-type]
    assert first == second


def test_a_fingerprint_with_a_missing_component_produces_no_key() -> None:
    """Treating an absent component as a value would merge every incomplete row into one hub."""
    frame = ieee_rows([{"account_id": "a"}, {"account_id": "b"}])
    frame.loc[frame.index[0], "id_33"] = None
    keys = fingerprint_keys(frame, FingerprintSpec("device_fp", ("DeviceInfo", "id_33")))
    assert pd.isna(keys.iloc[0])
    assert pd.notna(keys.iloc[1])


def test_circularity_with_the_account_uid_is_declared_per_fingerprint() -> None:
    """``account_id`` is built as ``c{card1}_a{addr1}_d{d1n}``, so a card fingerprint is
    partly a restatement of the identity it claims to link across."""
    by_name = {spec.name: spec for spec in IEEE_FINGERPRINTS}
    assert by_name["card_fp"].is_circular
    assert set(by_name["card_fp"].shared_with_uid) <= UID_COLUMNS
    assert not by_name["device_fp"].is_circular
    assert by_name["device_fp"] in NON_CIRCULAR_FINGERPRINTS
    assert all(not spec.is_circular for spec in NON_CIRCULAR_FINGERPRINTS)


# --- Ring flags ---------------------------------------------------------------------------


def test_flag_rings_reports_membership_and_the_metrics_that_drove_it() -> None:
    """The Phase 4 brief's required output shape."""
    snapshot = snapshot_of(chain_frame())
    flags = flag_rings(snapshot)  # type: ignore[arg-type]
    assert flags, "no ring was flagged on a fixture built to contain one"
    flag = flags[0]
    assert flag.size == len(flag.member_account_ids)
    assert flag.ring_risk_score is None, "an unfitted scorer must not invent a score"
    assert flag.betweenness_exact is True
    assert flag.snapshot_end == snapshot.snapshot_end  # type: ignore[attr-defined]


def test_build_score_table_keeps_an_accounts_strongest_ring() -> None:
    """An account in two communities is as risky as its riskiest one."""
    from app.models.tier3_graph import RingFlag

    moment = EPOCH + timedelta(hours=1)
    flags = [
        RingFlag(
            ring_id="r0",
            member_account_ids=("a", "b", "c"),
            size=3,
            ring_risk_score=0.2,
            density=1.0,
            max_degree_centrality=1.0,
            max_betweenness=0.0,
            betweenness_exact=True,
            chain_edge_count=0,
            entity_count=0,
            snapshot_end=moment,
        ),
        RingFlag(
            ring_id="r1",
            member_account_ids=("a", "d", "e"),
            size=3,
            ring_risk_score=0.9,
            density=1.0,
            max_degree_centrality=1.0,
            max_betweenness=0.0,
            betweenness_exact=True,
            chain_edge_count=0,
            entity_count=0,
            snapshot_end=moment,
        ),
    ]
    scores, ring_of, sizes = build_score_table(flags)
    assert scores["a"] == pytest.approx(0.9)
    assert ring_of["a"] == "r1"
    assert sizes["r0"] == 3


def test_fit_ring_scorer_refuses_a_single_class_training_set() -> None:
    """A constant scorer would rank everything equally and report a meaningless PR-AUC."""
    snapshot = snapshot_of(chain_frame())
    features = ring_feature_frame(snapshot, detect_communities(snapshot))  # type: ignore[arg-type]
    labels = np.ones(len(features), dtype=bool)
    with pytest.raises(ValueError, match="single class"):
        fit_ring_scorer(features, labels, "paysim")


def test_scorer_returns_probabilities_in_range() -> None:
    """``AuditRecord`` and the Phase 5 meta-learner both expect a bounded score."""
    snapshot = snapshot_of(chain_frame())
    features = ring_feature_frame(snapshot, detect_communities(snapshot))  # type: ignore[arg-type]
    labels = np.array([index % 2 == 0 for index in range(len(features))], dtype=bool)
    scorer = fit_ring_scorer(features, labels, "paysim")
    scores = scorer.score_frame(features)
    assert isinstance(scorer, RingScorer)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_scorer_refuses_a_frame_missing_a_feature() -> None:
    """A missing feature is an error, not a zero -- the rule Tier-1 states and Tier-2 repeats."""
    snapshot = snapshot_of(chain_frame())
    features = ring_feature_frame(snapshot, detect_communities(snapshot))  # type: ignore[arg-type]
    labels = np.array([index % 2 == 0 for index in range(len(features))], dtype=bool)
    scorer = fit_ring_scorer(features, labels, "paysim")
    with pytest.raises(KeyError):
        scorer.score_frame(features.drop(columns=["ring_density"]))


# --- Surrogate ground truth ----------------------------------------------------------------


def test_overlap_coefficient_matches_by_the_smaller_set() -> None:
    """Overlap, not Jaccard: a detected ring nested inside a larger true ring is a recovery."""
    assert driver.overlap_coefficient({"a", "b"}, {"a", "b", "c", "d"}) == pytest.approx(1.0)
    assert driver.overlap_coefficient({"a"}, {"b"}) == 0.0
    assert driver.overlap_coefficient(set(), {"a"}) == 0.0


def test_recovery_counts_each_true_ring_once() -> None:
    """Two detected fragments of one true ring are one recovery, not two."""
    detected = [{"a", "b", "c"}, {"a", "b", "d"}]
    truth = [{"a", "b", "c", "d"}]
    result = driver.measure_recovery(detected, truth, 0.5)
    assert result.matched_detected == 2
    assert result.matched_truth == 1
    assert result.recall == pytest.approx(1.0)


def test_surrogate_rings_are_built_from_fraud_rows_only() -> None:
    """The surrogate partition is derived from labels; nothing else may leak into it."""
    frame = chain_frame()
    rings = driver.surrogate_rings(frame, driver.paysim_factory(1, 0.0))
    members = {account for ring in rings for account in ring}
    assert "hub" not in members, "a clean star entered the surrogate ground truth"
    assert not any(account.startswith("s") for account in members)


def test_enrichment_needs_no_ground_truth_partition() -> None:
    """The robustness view must be computable from ring labels and scores alone."""
    rings = pd.DataFrame(
        {
            "is_fraud_ring": [True, True, False, False, False, False, False, False, False, False],
            "ring_risk_score": [0.99, 0.98, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.01],
        }
    )
    result = driver.measure_enrichment(rings, k=2)
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.base_rate == pytest.approx(0.2)
    assert result.lift == pytest.approx(5.0)


# --- Reproducibility -----------------------------------------------------------------------


def test_community_detection_is_deterministic() -> None:
    """Louvain is randomised; the seed is set and logged, so two runs must agree."""
    frame = chain_frame()
    first = detect_communities(snapshot_of(frame))  # type: ignore[arg-type]
    second = detect_communities(snapshot_of(frame))  # type: ignore[arg-type]
    assert first == second


def test_graph_feature_version_moves_with_the_parameters() -> None:
    """A past prediction must trace to the exact structural definition behind it."""
    one = driver.graph_feature_version("paysim", {"step_window": 1})
    same = driver.graph_feature_version("paysim", {"step_window": 1})
    other = driver.graph_feature_version("paysim", {"step_window": 3})
    assert one == same
    assert one != other
    assert one.startswith("gv_")


def test_a_significant_degradation_is_not_reported_as_a_win() -> None:
    """A bootstrap interval that excludes zero on the negative side is a loss, not a win.

    The obvious two-branch phrasing -- "straddles zero" against "excludes zero, therefore
    real" -- calls a significant degradation a success. Phase 4 measured exactly that case on
    IEEE-CIS (delta -0.0041, CI [-0.0048, -0.0034]), so the wire is here rather than in a
    reviewer's memory.
    """
    assert (
        driver.interval_verdict(-0.0041, -0.0048, -0.0034, gain="GAIN", loss="LOSS", tie="TIE")
        == "LOSS"
    )
    assert driver.interval_verdict(0.1, 0.05, 0.15, gain="GAIN", loss="LOSS", tie="TIE") == "GAIN"
    assert driver.interval_verdict(0.01, -0.02, 0.04, gain="GAIN", loss="LOSS", tie="TIE") == "TIE"
    assert driver.interval_verdict(0.0, 0.0, 0.0, gain="GAIN", loss="LOSS", tie="TIE") == "TIE"


def test_operating_point_reports_its_flag_rate_beside_its_precision(
    baseline_report: driver.CorpusReport,
) -> None:
    """Precision at a 40% flag rate and precision at a 0.5% flag rate are different products.

    ``models/README.md`` names `capacity_constrained_operating_point` as part of a complete
    registry entry for exactly this reason, so the driver has to produce one.
    """
    point = baseline_report.operating_point
    assert point, "no operating point was described"
    assert point["unit"] == baseline_report.unit
    assert 0.0 <= point["flag_rate"] <= 1.0
    assert "precision" in point and "recall" in point
    assert point["review_capacity_cap"] == driver.MAX_REVIEW_FLAG_RATE
    assert set(point["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


def test_cost_sensitivity_is_measured_not_promised(
    baseline_report: driver.CorpusReport,
) -> None:
    """ml-evaluation-standards section 3 asks for a +/-50% sweep wherever a threshold is
    recommended, and it has to be a computed sweep rather than a sentence in a docstring."""
    assert baseline_report.scale_sweep, "no +/-50% cost sensitivity sweep was produced"
    factors = {round(float(row.factor), 3) for row in baseline_report.scale_sweep}
    assert {0.5, 1.0, 1.5} <= factors
    assert baseline_report.cost_sweep, "no review-cost sweep was produced"


def test_a_leak_suspicious_result_carries_its_caveat_into_the_registry(
    tmp_path: Path,
) -> None:
    """A PR-AUC above the leak threshold must reach registry.json with the warning attached.

    BUILD_LOG Phase 2 records why: a PaySim PR-AUC of 0.9999 was once written to the registry
    with no warning on it, so anyone reading the registry alone would have taken it at face
    value. Phase 4's PaySim ring result is 0.9942 on a 0.767 base rate and trips the same wire.
    """
    from app.ml.evaluation import LEAK_SUSPICION_PR_AUC, evaluate

    labels = np.array([True] * 8 + [False] * 2)
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.02, 0.01])
    result = evaluate(
        "leaky",
        "test",
        labels,
        scores,
        threshold=0.5,
        threshold_criterion="fixture",
    )
    assert result.is_leak_suspicious, "fixture must be leak-suspicious to test the caveat"
    assert result.pr_auc > LEAK_SUSPICION_PR_AUC

    report = run_small_corpus(corpus_frame())
    report.ring_test = result
    model = driver.build_served_model(
        report,
        "tier3-graph-louvain-paysim-caveat-test",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    registry = tmp_path / "registry.json"
    registry.write_text("[]", encoding="utf-8")

    entry = driver.register(report, model, registry, tmp_path)
    assert entry.notes, "no notes were recorded at all"
    assert entry.notes[0].startswith(
        "DO NOT QUOTE AS A HEADLINE"
    ), "the leak caveat must be the first thing a registry reader sees"
    assert str(LEAK_SUSPICION_PR_AUC) in entry.notes[0]


# --- Regressions for the Phase 4 code review ------------------------------------------------


def test_a_chain_edge_does_not_delete_an_observed_flow_edge() -> None:
    """A mule is routinely both a flow endpoint and a chain endpoint.

    The first implementation wrote the chain edge's attribute dict over the flow edge's, so the
    observed edge vanished and the pair counted as pure chain structure -- inflating
    `ring_chain_edge_count` and `account_chain_degree`, both of which the scorer reads.
    """
    frame = paysim_rows(
        [
            {
                "account_id": "M",
                "counterparty_id": "X",
                "transaction_type": CASH_OUT,
                "amount": 100.0,
                "step": 5,
            },
            {
                "account_id": "V",
                "counterparty_id": "X",
                "transaction_type": TRANSFER,
                "amount": 100.0,
                "step": 5,
            },
        ]
    )
    snapshot = snapshot_of(frame, step_window=0)
    edge = snapshot.graph.get_edge_data("M", "X")  # type: ignore[attr-defined]
    assert edge is not None, "the observed flow edge M->X was deleted outright"
    assert edge["flow"] is True, "the flow relation was overwritten by the chain relation"
    assert edge["chain"] is True, "the inferred chain relation was not recorded"


def test_an_account_in_a_too_small_component_says_so_when_it_abstains() -> None:
    """The two abstention reasons must be distinguishable in an audit row.

    `ring_of` only ever holds scoreable accounts, so keying the reason off it made
    ABSTAIN_BELOW_MIN_RING unreachable and every abstention claimed the account had never been
    seen -- including accounts Tier-3 had seen and simply found no ring for.
    """
    model = fitted_model(chain_frame())
    seen_but_unscoreable = "s1"
    assert seen_but_unscoreable not in model.scores, "fixture account must not be scoreable"
    model.seen_accounts = frozenset({seen_but_unscoreable})

    result = model.score(transaction(seen_but_unscoreable))
    assert result.ring_risk_score is None
    assert result.abstention_reason == ABSTAIN_BELOW_MIN_RING

    unseen = model.score(transaction("never-seen-anywhere"))
    assert unseen.abstention_reason == ABSTAIN_NOT_IN_SNAPSHOT


def test_artifact_path_refuses_anything_that_leaves_the_directory(tmp_path: Path) -> None:
    """Separator checks alone are not enough on Windows, which this project is developed on.

    `Path("dir") / "C:secret.json"` evaluates to `C:secret.json` -- pathlib drops the base
    whenever a segment carries a drive letter -- so the id must be a single relative
    component and the resolved path is re-checked for containment.
    """
    good = artifact_path("tier3-graph-louvain-paysim-test", tmp_path, ".json")
    assert good.parent == tmp_path.resolve()
    for bad in ("C:secret", "C:/escape", "/etc/passwd", "../out", "a/b"):
        with pytest.raises(ValueError):
            artifact_path(bad, tmp_path, ".json")


def test_each_ring_is_counted_once_and_only_in_its_earliest_split() -> None:
    """Overlapping snapshot windows re-emit the same ring for several consecutive snapshots.

    Left alone, the scorer is selected and scored on rings it was fitted on, and the bootstrap
    treats near-identical copies as independent draws. Both are the correlated-positives
    artefact Phase 3 removed from Tier-2 by fixing its unit of analysis.
    """
    moments = pd.to_datetime(["2017-01-05", "2017-01-06", "2017-01-07"], utc=True)
    rings = pd.DataFrame(
        {
            "snapshot_end": [moments[0]] * 3 + [moments[1]] * 3 + [moments[2]] * 3,
            "ring_id": ["r0"] * 3 + ["r0"] * 3 + ["r1"] * 3,
            "account_id": ["a", "b", "c"] * 2 + ["x", "y", "z"],
            "split": ["train"] * 3 + ["val"] * 3 + ["test"] * 3,
        }
    )
    kept = driver.assign_rings_to_first_split(rings)
    surviving = set(zip(kept["snapshot_end"], kept["ring_id"], strict=True))
    assert (moments[0], "r0") in surviving, "the first sighting of a ring must be kept"
    assert (moments[1], "r0") not in surviving, "a ring the scorer trained on reappeared in val"
    assert (moments[2], "r1") in surviving, "a genuinely new ring was dropped"
    assert set(kept["split"]) == {"train", "test"}


def test_deduplication_guard_would_notice_if_it_stopped_working() -> None:
    """Guard on the guard: distinct rings must all survive, or the test above passes trivially."""
    moments = pd.to_datetime(["2017-01-05", "2017-01-06"], utc=True)
    rings = pd.DataFrame(
        {
            "snapshot_end": [moments[0]] * 3 + [moments[1]] * 3,
            "ring_id": ["r0"] * 3 + ["r0"] * 3,
            "account_id": ["a", "b", "c", "p", "q", "r"],
            "split": ["train"] * 3 + ["val"] * 3,
        }
    )
    kept = driver.assign_rings_to_first_split(rings)
    assert len(kept) == 6, "two different rings that share a ring_id were collapsed"


def test_ring_exposure_counts_a_within_ring_transfer_once() -> None:
    """Summing both sides of every edge priced a ring's own transfer at twice its amount.

    Every chain-linked PaySim ring is exactly that shape, and the amount feeds the
    false-negative side of the cost model, so the doubling biased the recommended threshold.
    """
    window = paysim_rows(
        [
            {
                "account_id": "V",
                "counterparty_id": "M",
                "transaction_type": TRANSFER,
                "amount": 100.0,
                "step": 1,
            }
        ]
    )
    exposure = driver.ring_exposure(window, [["V", "M", "Z"]])
    assert exposure["r0"] == pytest.approx(100.0), "the transaction was counted more than once"


def test_graph_feature_version_ignores_the_fitted_threshold() -> None:
    """`feature_version` identifies inputs. A threshold is an output of fitting."""
    structural = {"max_entity_degree": 200, "cadence_hours": 168.0}
    assert driver.graph_feature_version("ieee_cis", structural) == driver.graph_feature_version(
        "ieee_cis", dict(structural)
    )
    with_threshold = {**structural, "operating_threshold": 0.71}
    assert driver.graph_feature_version("ieee_cis", with_threshold) != (
        driver.graph_feature_version("ieee_cis", structural)
    ), "the helper must still hash whatever it is given -- the fix is what register() passes"


def test_registry_feature_version_is_stable_across_thresholds(
    baseline_report: driver.CorpusReport, tmp_path: Path
) -> None:
    """Two models over identical graphs must carry the same `gv_` hash."""
    model = driver.build_served_model(
        baseline_report,
        "tier3-graph-louvain-paysim-fv-a",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    registry = tmp_path / "registry.json"
    registry.write_text("[]", encoding="utf-8")
    first = driver.register(baseline_report, model, registry, tmp_path)

    shifted = replace(baseline_report, threshold=baseline_report.threshold * 0.5 + 0.1)
    other = driver.build_served_model(
        shifted,
        "tier3-graph-louvain-paysim-fv-b",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    second = driver.register(shifted, other, registry, tmp_path)

    assert first.feature_version == second.feature_version
    assert first.hyperparameters["operating_threshold"] != (
        second.hyperparameters["operating_threshold"]
    ), "the threshold must still be recorded, just not hashed"


def test_training_window_describes_the_train_split_only(
    baseline_report: driver.CorpusReport, tmp_path: Path
) -> None:
    """An auditor tracing a prediction must not read a window that includes held-out data."""
    model = driver.build_served_model(
        baseline_report,
        "tier3-graph-louvain-paysim-window",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    registry = tmp_path / "registry.json"
    registry.write_text("[]", encoding="utf-8")
    entry = driver.register(baseline_report, model, registry, tmp_path)

    frame = corpus_frame()
    train_end = frame.loc[frame["split"] == "train", "event_time"].max()
    assert entry.training_window["end"] == str(train_end)
    assert pd.Timestamp(entry.training_window["end"]) < frame["event_time"].max()


def test_ring_metrics_use_a_threshold_chosen_on_rings(
    baseline_report: driver.CorpusReport,
) -> None:
    """A ring confusion matrix cut at a transaction-selected threshold reports an operating
    point that was never chosen for that unit."""
    assert baseline_report.ring_test is not None
    assert "ring" in baseline_report.ring_test.threshold_criterion
    assert "transaction" not in baseline_report.ring_test.threshold_criterion
    assert baseline_report.ring_test.threshold == pytest.approx(baseline_report.ring_threshold)


def test_a_below_no_skill_headline_carries_a_caveat_into_the_registry(
    baseline_report: driver.CorpusReport, tmp_path: Path
) -> None:
    """Leak suspicion is not the only way a headline fails its bar.

    A PR-AUC under its own no-skill floor is worse than a coin toss, and it reached the
    rendered report but not registry.json -- which is the audit artefact.
    """
    from app.ml.evaluation import evaluate

    # Deliberately anti-correlated: the positives are ranked last, so the ranking is worse
    # than random and PR-AUC falls below the base rate.
    labels = np.array([False] * 27 + [True] * 3)
    scores = np.arange(30, 0, -1, dtype="float64")
    weak = evaluate("weak", "test", labels, scores, threshold=0.5, threshold_criterion="fixture")
    assert weak.lift_over_no_skill < 1.0, "fixture must be below no-skill to test the caveat"

    report = replace(baseline_report, ring_test=weak)
    model = driver.build_served_model(
        report,
        "tier3-graph-louvain-paysim-weak",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    registry = tmp_path / "registry.json"
    registry.write_text("[]", encoding="utf-8")
    entry = driver.register(report, model, registry, tmp_path)
    assert any(note.startswith("BELOW NO-SKILL") for note in entry.notes)


def test_a_negative_fusion_delta_carries_a_caveat_into_the_registry(
    baseline_report: driver.CorpusReport, tmp_path: Path
) -> None:
    """The measured IEEE-CIS case: CI [-0.0038, -0.0025], excluding zero on the wrong side."""
    report = replace(
        baseline_report,
        tier1_pr_auc=0.5276,
        fused_pr_auc=0.5245,
        fused_delta=(-0.003084, -0.003756, -0.002469),
    )
    model = driver.build_served_model(
        report,
        "tier3-graph-louvain-paysim-fusion",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    registry = tmp_path / "registry.json"
    registry.write_text("[]", encoding="utf-8")
    entry = driver.register(report, model, registry, tmp_path)
    assert any(note.startswith("NEGATIVE FUSION DELTA") for note in entry.notes)


def test_offline_and_served_score_tables_agree(
    baseline_report: driver.CorpusReport,
) -> None:
    """The offline metric and the served table must not drift apart.

    `attach_scores` takes a vectorised max over rings; the served table is built by
    `build_score_table` from RingFlags. Three hand-written copies of that rule used to exist
    and the only thing asserting they agreed was a comment.
    """
    account_rings = baseline_report.winner.account_rings
    latest = account_rings["snapshot_end"].max()
    current = account_rings.loc[account_rings["snapshot_end"] == latest]

    model = driver.build_served_model(
        baseline_report,
        "tier3-graph-louvain-paysim-agree",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    offline = current.groupby("account_id")["ring_risk_score"].max().to_dict()

    assert set(model.scores) == {str(k) for k in offline}
    for account, value in offline.items():
        assert model.scores[str(account)] == pytest.approx(float(value))


def test_latency_benchmark_reports_the_tail(baseline_report: driver.CorpusReport) -> None:
    """p99 is the figure that decides whether Phase 7's Tier-3 timeout is implementable, and
    `models/README.md` names it as part of a complete entry."""
    model = driver.build_served_model(
        baseline_report,
        "tier3-graph-louvain-paysim-latency",
        corpus_frame(),
        driver.paysim_factory(0, 0.0),
        entity_anonymization_key=TEST_ENTITY_ANONYMIZATION_KEY,
    )
    account = next(iter(model.scores))
    measured = benchmark_latency(model, [transaction(account)], calls=20)
    assert {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"} <= set(measured)
    assert measured["p50_ms"] <= measured["p95_ms"] <= measured["p99_ms"] <= measured["max_ms"]


def test_surrogate_recovery_never_matches_across_snapshots() -> None:
    """Consecutive windows share most of their accounts, so a pooled match lets a ring from one
    window recover a surrogate ring from another and inflates recovery precision."""
    detected = [{"a", "b", "c"}]
    truth = [{"a", "b", "c"}]
    within = driver.measure_recovery(detected, truth, 0.5)
    assert within.matched_detected == 1

    unrelated = driver.measure_recovery(detected, [{"x", "y", "z"}], 0.5)
    assert unrelated.matched_detected == 0
    assert unrelated.precision == 0.0


@pytest.mark.parametrize("model_id", ["C:secret", "C:/escape", "/etc/passwd", "../out", "a/b"])
def test_tier2_load_refuses_the_same_traversing_ids(model_id: str, tmp_path: Path) -> None:
    """Tier-2 carried the identical hole, so the containment check is shared, not copied.

    A guard that lives in each tier is a guard each new tier can forget, which is how the same
    Windows drive-relative traversal ended up in two of them.
    """
    from app.models.tier2_behavioral import Tier2Model

    with pytest.raises(ValueError):
        Tier2Model.load(model_id, tmp_path)


def test_ring_metrics_use_a_ring_threshold_even_when_the_unit_is_the_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch a PaySim-only fixture cannot reach.

    On PaySim the reported threshold and the ring threshold coincide, because the ring *is* the
    unit, so a test built on it passes whether or not the ring metric was cut at a
    transaction-selected operating point. IEEE-CIS keeps the transaction as its unit, and there
    the two differ -- which is exactly where the first attempt at this fix silently failed to
    apply and shipped a ring confusion matrix labelled "chosen on validation transactions".

    Forcing the coverage floor is the honest way to reach that branch from a small fixture:
    it changes which unit the corpus reports, and nothing about the graph or the scoring.
    """
    monkeypatch.setattr(driver, "TRANSACTION_COVERAGE_FLOOR", 2.0)
    report = run_small_corpus(corpus_frame())

    assert report.unit == "transaction", "the coverage floor did not force the branch"
    assert report.ring_test is not None
    # "ring" and never "transaction": the small fixture can land on the degenerate branch
    # ("no finite threshold exists: every validation ring abstained"), which is still a ring
    # criterion. What must never appear is the transaction wording the bug produced.
    criterion = report.ring_test.threshold_criterion
    assert "ring" in criterion, "ring metric cut at a threshold from another unit"
    assert "transaction" not in criterion, criterion
    assert report.ring_test.threshold == pytest.approx(report.ring_threshold)


def test_the_visualisation_actually_finds_the_chain_edges_it_promises(tmp_path: Path) -> None:
    """The legend claims a thick red edge marks an inferred chain link.

    When edges stopped carrying a `relation` label, the plot's filter kept matching nothing and
    every chain link was drawn as an ordinary grey edge -- a picture that quietly contradicted
    its own caption, which no metric would ever have caught.
    """
    frame = chain_frame()
    snapshot = snapshot_of(frame)
    chain_edges = [
        (u, v) for u, v, d in snapshot.graph.edges(data=True) if d.get("chain")  # type: ignore[attr-defined]
    ]
    assert chain_edges, "fixture must contain a chain edge for this test to mean anything"
    assert not any(
        "relation" in d for _, _, d in snapshot.graph.edges(data=True)  # type: ignore[attr-defined]
    ), "a stale `relation` attribute is still being written"

    members = sorted({node for edge in chain_edges for node in edge})
    destination = tmp_path / "ring.png"
    driver.plot_ring(cast(GraphSnapshot, snapshot), members, set(), "fixture", destination)
    assert destination.exists() and destination.stat().st_size > 0


def test_no_tier_builds_an_artefact_path_by_string_concatenation() -> None:
    """The guard is only worth having if every call site uses it.

    A security review found `load_tier1_scores` building `artifact_dir / f"{model_id}.txt"`
    by hand from a `model_id` read out of `registry.json` — file content, which
    `append_entry` does not re-validate — in the same change that added the guard to close
    exactly that hole. A guard one call site can skip is how the bug survives its own fix, so
    this asserts the pattern is absent repo-wide rather than trusting each author to remember.
    """
    import re

    source_root = Path(__file__).resolve().parents[1] / "app"
    offenders: list[str] = []
    pattern = re.compile(r"(?:directory|artifact_dir)\s*/\s*f\"\{(?:self\.)?model_id\}")
    for path in source_root.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(source_root)}:{number}")
    assert not offenders, (
        "these build an artefact path from a model id by concatenation instead of going "
        f"through app.ml.registry.artifact_path: {offenders}"
    )


def test_near_duplicate_rings_are_de_duplicated_not_just_exact_ones() -> None:
    """One member changing does not make a ring a different ring.

    The first de-duplication keyed on the exact member set. Measured on the real corpora, 58%
    of the surviving PaySim test rings and 82% of the IEEE-CIS validation rings still
    overlapped a training-split ring by at least half — so the scorer was still being selected
    and scored on rings it had been fitted on, and the bootstrap was still resampling
    near-identical copies as independent draws.
    """
    moments = pd.to_datetime(["2017-01-05", "2017-01-06", "2017-01-07"], utc=True)
    rings = pd.DataFrame(
        {
            "snapshot_end": [moments[0]] * 4 + [moments[1]] * 4 + [moments[2]] * 4,
            "ring_id": ["r0"] * 4 + ["r0"] * 4 + ["r1"] * 4,
            # Second ring shares three of four members with the first: a different exact set,
            # the same ring. Third shares nothing.
            "account_id": ["a", "b", "c", "d"] + ["a", "b", "c", "e"] + ["w", "x", "y", "z"],
            "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
        }
    )
    kept = driver.assign_rings_to_first_split(rings)
    surviving = set(zip(kept["snapshot_end"], kept["ring_id"], strict=True))
    assert (moments[0], "r0") in surviving, "the first sighting must be kept"
    assert (
        moments[1],
        "r0",
    ) not in surviving, (
        "a ring sharing three of four members with a training ring was treated as new"
    )
    assert (moments[2], "r1") in surviving, "a genuinely disjoint ring was dropped"


def test_the_unit_of_analysis_is_selected_on_validation_not_test() -> None:
    """Which unit a corpus is evaluated on is a selection like any other.

    It used to be decided from the test split's abstention rate. The existing threshold guard
    could not catch it: that guard perturbs test *labels*, and the abstention rate depends on
    NaN *scores*, so it was invariant to the corruption and passed while the contamination
    stood.
    """
    import inspect

    source = inspect.getsource(driver.run_corpus)
    assert (
        "ring_unit = validation_abstention >= TRANSACTION_COVERAGE_FLOOR" in source
    ), "the unit of analysis is being chosen from something other than validation abstention"
    assert (
        "abstention = abstention_rate(test_txns)" in source
    ), "the test abstention rate must still be measured, for the reader to interpret"


def test_ring_costs_are_labelled_per_ring_not_per_transaction() -> None:
    """A ring cost divided by a ring count but labelled per-transaction reads ~60x wrong."""
    labels = np.array([True, False, True, False])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])
    estimate = cost_at_threshold(labels, scores, amounts, 0.5, driver.ring_cost_model(CostModel()))
    rendered = estimate.render()
    assert "per 1,000 rings" in rendered, rendered
    assert "per 1,000 transactions" not in rendered
    payload = estimate.to_dict()
    assert payload["unit"] == "ring"
    assert "cost_per_1000_rings" in payload


def test_uniform_cost_scaling_cannot_move_the_threshold() -> None:
    """Pinned because the report now says so, and a reader must be able to trust that.

    Scaling both cost parameters by the same factor multiplies total cost by that factor, so
    the argmin is unchanged by construction. The sweep is reported because section 3 names a
    +/-50% analysis; its flatness is arithmetic, not evidence of robustness.
    """
    from app.ml.cost import sensitivity_sweep

    labels = np.array([True] * 30 + [False] * 70)
    scores = np.linspace(0.0, 1.0, 100)
    amounts = np.full(100, 50.0)
    rows = sensitivity_sweep(labels, scores, amounts, CostModel())
    thresholds = {round(row.threshold, 9) for row in rows}
    assert len(thresholds) == 1, "if this ever varies, the report's claim needs rewriting"
    costs = {round(row.factor, 3): row.estimate.total_cost for row in rows}
    assert costs[1.5] == pytest.approx(costs[1.0] * 1.5)
    assert costs[0.5] == pytest.approx(costs[1.0] * 0.5)


def test_community_detection_is_deterministic_across_processes() -> None:
    """The in-process determinism test cannot see the bug that actually bit.

    CPython randomises string hashing per process, and both `connected_components` and Louvain
    return *sets*, so node order — and with it Louvain's partition of larger components — used
    to change between runs while being perfectly stable within one. Two runs of the same code
    at the same seed produced ring PR-AUC 0.986566 and 0.986451. Calling the function twice in
    one interpreter, as the sibling test does, is blind to that; this spawns real subprocesses.
    """
    import subprocess
    import sys

    script = (
        "import pandas as pd\n"
        "from datetime import datetime, timedelta, UTC\n"
        "from app.models.tier3_graph import PaySimMoneyFlowGraph, detect_communities\n"
        "E = datetime(2017, 1, 1, tzinfo=UTC)\n"
        "rows = []\n"
        "for i in range(40):\n"
        "    rows.append({'account_id': f'v{i}', 'counterparty_id': f'm{i%7}',\n"
        "                 'transaction_type': 'TRANSFER', 'amount': 100.0 + i, 'step': 5})\n"
        "    rows.append({'account_id': f'o{i}', 'counterparty_id': f'd{i%5}',\n"
        "                 'transaction_type': 'CASH_OUT', 'amount': 100.0 + i, 'step': 5})\n"
        "f = pd.DataFrame(rows)\n"
        "f['event_time'] = [E + timedelta(hours=4)] * len(f)\n"
        "f['transaction_id'] = [str(i) for i in range(len(f))]\n"
        "f['is_fraud'] = False\n"
        "g = PaySimMoneyFlowGraph(step_window=0, amount_tolerance=0.0)\n"
        "g.insert(f)\n"
        "print(detect_communities(g.snapshot(E + timedelta(hours=9))))\n"
    )
    root = Path(__file__).resolve().parents[1]
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(3)
    }
    assert len(outputs) == 1, (
        "community detection differs between processes; a set is being iterated somewhere "
        "without an explicit ordering"
    )


def test_de_duplication_does_not_reach_the_serving_score_table(
    baseline_report: driver.CorpusReport,
) -> None:
    """De-duplication is a metric device; the score table is a serving artefact.

    Applying it to both made IEEE-CIS's transaction abstention rate jump from 65.7% to 96.7% —
    a fact about the evaluation's own bookkeeping, reported as a property of the layer. In
    production every ring in the current snapshot is scored, repeats included, and an account
    whose only ring happens to resemble an earlier one is still an account sitting in a ring.
    """
    account_rings = baseline_report.winner.account_rings
    evaluation_rings = baseline_report.winner.rings

    scored_pairs = set(zip(account_rings["snapshot_end"], account_rings["ring_id"], strict=True))
    evaluated_pairs = set(
        zip(evaluation_rings["snapshot_end"], evaluation_rings["ring_id"], strict=True)
    )
    assert evaluated_pairs <= scored_pairs, "the metric population is not a subset of the scored"
    assert len(scored_pairs) >= len(evaluated_pairs), (
        "the served table holds fewer rings than the metric does; de-duplication has leaked "
        "into the scoring path"
    )


def test_recovery_and_the_ring_metric_read_the_same_population(
    baseline_report: driver.CorpusReport,
) -> None:
    """One entry, one population. Recovery is a ring-level metric like the ring PR-AUC.

    Scoping de-duplication to the evaluation set fixed the abstention rate but pointed recovery
    back at the full scored population, so the registry reported 18,839 detected rings beside a
    ring test computed over 2,508 — two different denominators in one record.
    """
    evaluation = baseline_report.winner.evaluation_account_rings
    full = baseline_report.winner.account_rings
    ring_view = baseline_report.winner.rings

    evaluation_pairs = set(zip(evaluation["snapshot_end"], evaluation["ring_id"], strict=True))
    view_pairs = set(zip(ring_view["snapshot_end"], ring_view["ring_id"], strict=True))
    full_pairs = set(zip(full["snapshot_end"], full["ring_id"], strict=True))

    assert evaluation_pairs == view_pairs, (
        "the account-level evaluation rings and the ring-level view disagree about which rings "
        "are being measured"
    )
    assert evaluation_pairs <= full_pairs
