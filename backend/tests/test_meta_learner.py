"""Meta-learner tests.

The meta-learner fuses three layers that were all fitted on the train split, so almost every
way this phase can go wrong produces a *better* number rather than a worse one. These tests are
therefore weighted towards the guards, and every guard is paired with a test that plants a
violation and asserts it fires -- a guard that never fires is indistinguishable from a guard
that cannot fire.

The ones that matter most:

- :func:`test_fold_ordering_guard_rejects_a_shuffled_assignment`, the companion to
  :func:`test_out_of_fold_folds_are_forward_chaining`. Random K-fold over transaction data lets
  a fold's model train on rows chronologically after the rows it scores. Nothing downstream
  would reveal it; the PR-AUC simply comes out too high.
- :func:`test_label_derived_columns_never_reach_the_meta_matrix`. Tier-3's snapshot frame
  carries ``account_is_fraudulent``, computed as ``fraud_accounts(window_rows)`` -- a direct
  label read -- in the same frame as the topology features.
- :func:`test_abstention_is_a_sentinel_plus_an_indicator_never_a_zero`. A ``0.0`` in a tier
  column reads to a tree as "this layer looked and found nothing wrong", about a layer that did
  not look. Phases 3 and 4 both deferred this distinction to here.
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.audit import AuditRecord
from app.data.schema import TransactionFeatures
from app.ml.cost import CostModel
from app.ml.evaluation import pr_auc
from app.models.meta_features import (
    ABSTENTION_SENTINEL,
    EXEMPT_BLOCKS,
    FEATURE_BLOCKS,
    META_DENIED_COLUMNS,
    EmpiricalCdf,
    RegisteredTier1,
    assert_forward_chaining,
    block_feature_names,
    build_oof_tier1,
    meta_denied_columns_present,
    rank_normalise,
    require_clean_feature_names,
    require_ieee_cis,
    time_blocks,
)
from app.models.meta_learner import (
    ABLATION_MAX_DEPTH,
    ABLATION_NUM_ROUNDS,
    MetaInputSpec,
    MetaModel,
    fit_booster,
    fit_calibrator,
)
from app.models.tier1_anomaly import Tier1Result
from app.models.tier2_behavioral import Tier2Result
from app.models.train_meta_learner import (
    BLOCK_DISABLED,
    AblationDelta,
    BlockVerdict,
    choose_operating_points,
    load_splits,
    validation_slices,
)

ROWS = 1_200


@pytest.fixture
def event_time() -> pd.Series:
    """Return a time-ordered event-time column with no ties."""
    start = datetime(2018, 1, 1, tzinfo=UTC)
    return pd.Series([start + timedelta(minutes=10 * index) for index in range(ROWS)])


# --- The fold scheme ---------------------------------------------------------------------


def test_blocks_partition_every_row_exactly_once(event_time: pd.Series) -> None:
    """Every row lands in exactly one block, and the blocks are contiguous in time."""
    blocks = time_blocks(event_time, block_count=5)
    assert blocks.block_count == 5
    assert set(np.unique(blocks.index)) == {1, 2, 3, 4, 5}
    assert len(blocks.index) == ROWS

    # Contiguity: block number is non-decreasing once rows are put in time order.
    ordered = blocks.index[event_time.to_numpy().argsort(kind="stable")]
    assert (np.diff(ordered) >= 0).all()


def test_blocks_are_close_to_equal_row_counts(event_time: pd.Series) -> None:
    """The cut is by row count, so blocks come out within a row of each other."""
    blocks = time_blocks(event_time, block_count=5)
    counts = np.bincount(blocks.index, minlength=6)[1:]
    assert counts.max() - counts.min() <= 1


def test_out_of_fold_folds_are_forward_chaining(event_time: pd.Series) -> None:
    """Each fold's fitting rows all precede every row that fold scores."""
    blocks = time_blocks(event_time, block_count=5)
    for scored in range(2, 6):
        assert_forward_chaining(blocks, event_time, scored)

        fit_mask = blocks.rows_before(scored)
        score_mask = blocks.rows_in(scored)
        assert fit_mask.sum() > 0
        # The two sets are disjoint: no row is ever used to fit the model that scores it.
        assert not (fit_mask & score_mask).any()
        assert event_time.to_numpy()[fit_mask].max() < event_time.to_numpy()[score_mask].min()


def test_fold_ordering_guard_rejects_a_shuffled_assignment(event_time: pd.Series) -> None:
    """A guard on the guard: random K-fold must be rejected, not silently accepted.

    This is the contamination out-of-fold scoring exists to prevent, and the one that no
    downstream metric would reveal -- a shuffled assignment simply reports a higher PR-AUC.
    """
    blocks = time_blocks(event_time, block_count=5)
    shuffled = np.random.default_rng(0).permutation(blocks.index)
    scrambled = type(blocks)(
        index=shuffled, boundaries=blocks.boundaries, block_count=blocks.block_count
    )
    with pytest.raises(ValueError, match="not forward-chaining"):
        assert_forward_chaining(scrambled, event_time, scored_block=3)


def test_block_one_cannot_be_scored_out_of_fold(event_time: pd.Series) -> None:
    """Block 1 has no predecessor, so asking for its out-of-fold score is an error.

    It is dropped from the meta-fit set rather than sentinel-filled: a sentinel there would
    teach the meta-learner a serving state that never occurs, since Tier-1 is never absent.
    """
    blocks = time_blocks(event_time, block_count=5)
    assert not blocks.rows_before(1).any()
    with pytest.raises(ValueError, match="no earlier block"):
        assert_forward_chaining(blocks, event_time, scored_block=1)


def test_rows_sharing_a_boundary_timestamp_fall_on_the_later_side() -> None:
    """Ties go to the later block, so no instant straddles a fit set and the set it scores."""
    start = datetime(2018, 1, 1, tzinfo=UTC)
    # 40 rows across only 4 distinct timestamps, so every boundary lands on a tie.
    times = pd.Series([start + timedelta(hours=index // 10) for index in range(40)])
    blocks = time_blocks(times, block_count=2)
    for moment in times.unique():
        assigned = set(blocks.index[times.to_numpy() == moment])
        assert len(assigned) == 1, f"timestamp {moment} was split across blocks {assigned}"


def test_block_cut_refuses_an_impossible_request() -> None:
    """Too few distinct timestamps to cut is an error, not a silently shrunken fit set."""
    start = datetime(2018, 1, 1, tzinfo=UTC)
    times = pd.Series([start] * 20)
    with pytest.raises(ValueError, match="empty"):
        time_blocks(times, block_count=5)


def test_block_cut_rejects_null_timestamps(event_time: pd.Series) -> None:
    """A null event time makes the cut undefined."""
    holed = event_time.copy()
    holed.iloc[5] = pd.NaT
    with pytest.raises(ValueError, match="nulls"):
        time_blocks(holed, block_count=5)


# --- Rank normalisation ------------------------------------------------------------------


def test_rank_normalisation_preserves_ordering() -> None:
    """Ranking is monotone, so it cannot change what a fold model asserted."""
    scores = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    ranked = rank_normalise(scores)
    assert list(ranked.argsort()) == list(scores.argsort())
    assert ranked.min() == 0.0
    assert ranked.max() == 1.0


def test_rank_normalisation_averages_ties() -> None:
    """Equal scores stay equal, rather than becoming an arbitrary order."""
    ranked = rank_normalise(np.array([0.5, 0.5, 0.5, 0.9]))
    assert ranked[0] == ranked[1] == ranked[2]
    assert ranked[3] > ranked[0]


def test_rank_normalisation_removes_a_level_shift() -> None:
    """Two folds differing only by a monotone recalibration produce the same ranks.

    This is the property the scheme relies on: fold models fitted on one, two, three and four
    blocks differ in level and sharpness, and that drift is correlated with time.
    """
    base = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    shifted = base * 0.4 + 0.2
    assert np.allclose(rank_normalise(base), rank_normalise(shifted))


def test_empirical_cdf_maps_the_reference_sample_onto_the_unit_interval() -> None:
    """The CDF fitted on train maps train scores roughly uniformly onto [0, 1]."""
    rng = np.random.default_rng(0)
    reference = rng.beta(2.0, 5.0, size=10_000)
    cdf = EmpiricalCdf.fit(reference)
    mapped = cdf.apply(reference)
    assert 0.0 <= mapped.min() and mapped.max() <= 1.0
    # Roughly uniform: deciles land where deciles should.
    for quantile in (0.25, 0.5, 0.75):
        assert abs(float(np.quantile(mapped, quantile)) - quantile) < 0.02


def test_empirical_cdf_is_monotone_and_round_trips() -> None:
    """The map preserves ordering, and survives the sidecar round trip unchanged."""
    reference = np.linspace(0.0, 1.0, 500)
    cdf = EmpiricalCdf.fit(reference)
    probe = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    mapped = cdf.apply(probe)
    assert list(mapped.argsort()) == list(probe.argsort())
    assert np.allclose(EmpiricalCdf.from_dict(cdf.to_dict()).apply(probe), mapped)


# --- The deny list -----------------------------------------------------------------------


def test_label_derived_columns_never_reach_the_meta_matrix() -> None:
    """The Tier-3 snapshot frame's label reads are barred from the feature set.

    ``account_is_fraudulent`` is ``fraud_accounts(window_rows)`` -- a direct label read -- and
    travels in the same frame as the topology features that Phase 5 does want.
    """
    for column in ("account_is_fraudulent", "is_fraud_ring", "ring_amount", "is_fraud"):
        assert column in META_DENIED_COLUMNS

    every_name = [name for names in FEATURE_BLOCKS.values() for name in names]
    assert meta_denied_columns_present(every_name) == []


def test_deny_list_guard_actually_catches_a_planted_leak() -> None:
    """The paired half: planting a denied column must raise, naming it."""
    planted = [*FEATURE_BLOCKS["engineered"], "account_is_fraudulent"]
    with pytest.raises(ValueError, match="account_is_fraudulent"):
        require_clean_feature_names(planted)


def test_deny_list_covers_identifiers_and_amount_duplicates() -> None:
    """The meta deny list is a superset of Tier-1's, not a replacement for it."""
    for column in ("account_id", "transaction_id", "event_time", "amount", "split"):
        assert column in META_DENIED_COLUMNS


def test_block_feature_names_are_order_independent() -> None:
    """The same blocks in a different order give the same matrix, hence the same version hash."""
    first = block_feature_names(["tier1", "engineered", "tier2"])
    second = block_feature_names(["tier2", "engineered", "tier1"])
    assert first == second


def test_block_feature_names_rejects_an_unknown_block() -> None:
    """A typo in a block name is an error, not a silently smaller feature set."""
    with pytest.raises(KeyError, match="tier4"):
        block_feature_names(["engineered", "tier4"])


def test_tier2_anomaly_flag_is_not_a_feature() -> None:
    """The registered Tier-2 threshold is -1.0, so its flag is constant and carries nothing."""
    assert "tier2_is_anomaly" not in FEATURE_BLOCKS["tier2"]


def test_tier3_blocks_separate_servable_from_unservable_features() -> None:
    """Topology is its own block because ``Tier3Result`` cannot supply it today.

    Keeping them separable is what turns "we trained on a feature we cannot serve" from an
    integration surprise into a costed, recorded Phase 7 requirement.
    """
    served = set(FEATURE_BLOCKS["tier3_served"])
    topology = set(FEATURE_BLOCKS["tier3_topology"])
    assert served.isdisjoint(topology)
    assert "tier3_ring_risk_score" in served
    assert "tier3_account_betweenness" in topology


def test_every_tier_block_carries_a_scoreability_indicator() -> None:
    """A tier that can abstain must be able to say so separately from its score."""
    assert "tier2_is_scoreable" in FEATURE_BLOCKS["tier2"]
    assert "tier3_is_scoreable" in FEATURE_BLOCKS["tier3_served"]


def test_abstention_is_a_sentinel_plus_an_indicator_never_a_zero() -> None:
    """The sentinel ranks below every real score and is not confusable with a real one.

    Tier-2's reconstruction error and Tier-3's ring risk are both non-negative, so a negative
    sentinel cannot collide with a genuine measurement, and it sorts last -- an abstaining layer
    never flags.
    """
    assert ABSTENTION_SENTINEL == -1.0
    assert ABSTENTION_SENTINEL < 0.0


def test_exempt_blocks_are_the_base_not_the_tiers_under_test() -> None:
    """Only the base blocks are exempt from retirement; Tier-2 and Tier-3 must be at risk."""
    assert EXEMPT_BLOCKS == {"engineered", "tier1"}
    for block in ("tier2", "tier3_served", "tier3_topology"):
        assert block not in EXEMPT_BLOCKS


# --- Corpus discipline -------------------------------------------------------------------


def test_meta_learner_refuses_paysim() -> None:
    """PaySim is barred by measurement: its Tier-1 result is a simulator artefact."""
    with pytest.raises(ValueError, match="simulator artefact"):
        require_ieee_cis("paysim")


def test_meta_learner_accepts_ieee_cis() -> None:
    """The guard is not simply always-raising."""
    require_ieee_cis("ieee_cis")


# --- Out-of-fold Tier-1 scoring ----------------------------------------------------------


def _synthetic_frame(rows: int = 2_000, seed: int = 7) -> pd.DataFrame:
    """Build a frame shaped like an IEEE-CIS parquet split, with a learnable signal.

    Synthetic rather than sampled from ``data/processed`` so the suite runs without the corpora
    present. The same shape ``test_tier1`` uses, so the Tier-1 input spec fits it unchanged.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2018, 1, 1, tzinfo=UTC)
    device_is_new = rng.random(rows) < 0.4
    amount = rng.lognormal(mean=4.0, sigma=1.0, size=rows)
    risk = 0.02 + 0.35 * device_is_new * (amount > np.quantile(amount, 0.8))
    is_fraud = rng.random(rows) < risk
    return pd.DataFrame(
        {
            "TransactionID": np.arange(rows, dtype="int64"),
            "TransactionDT": np.arange(rows, dtype="int64") * 600,
            "transaction_id": [str(index) for index in range(rows)],
            "source_dataset": "ieee_cis",
            "event_time": [start + timedelta(minutes=10 * index) for index in range(rows)],
            "amount": amount,
            "amount_log": np.log1p(amount),
            "account_id": [f"c{index % 200}" for index in range(rows)],
            "counterparty_id": pd.Series([None] * rows, dtype="object"),
            "ProductCD": pd.Series(rng.choice(["W", "C", "R"], rows), dtype="object"),
            "card4": pd.Series(rng.choice(["visa", "mastercard"], rows), dtype="object"),
            "DeviceInfo": pd.Series(
                rng.choice([f"dev{index}" for index in range(200)], rows), dtype="object"
            ),
            "C1": rng.integers(0, 40, rows).astype("float64"),
            "D1": rng.integers(0, 400, rows).astype("float64"),
            "hour_of_day": rng.integers(0, 24, rows).astype("int16"),
            "device_is_new": device_is_new,
            "velocity_count_24h": rng.integers(1, 12, rows).astype("float64"),
            "velocity_available": True,
            "is_fraud": is_fraud,
            "split": "train",
            "feature_version": "fv_synthetic",
            "uid_strategy": "card_addr_d1n",
        }
    )


class _StubTier1Model:
    """A stand-in for the registered full-train model, for the reference distribution alone.

    ``build_oof_tier1`` uses the registered model only to fit the reference CDF and to compute
    the handicap yardstick; neither needs a real booster, and loading a 13MB LightGBM artefact
    into a unit test would tie the suite to ``models/artifacts`` being present.
    """

    def __init__(self, seed: int = 3) -> None:
        self._rng = np.random.default_rng(seed)

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a score correlated with the planted signal, so PR-AUC is meaningful."""
        base = frame["device_is_new"].to_numpy(dtype="float64") * 0.5
        return np.clip(base + self._rng.random(len(frame)) * 0.5, 0.0, 1.0)


@pytest.fixture
def oof_splits() -> dict[str, pd.DataFrame]:
    """Return a chronological train/validation pair from the synthetic frame."""
    frame = _synthetic_frame()
    cut = int(0.8 * len(frame))
    return {"train": frame.iloc[:cut].copy(), "val": frame.iloc[cut:].copy()}


@pytest.fixture
def registered_stub() -> RegisteredTier1:
    """Return a registered-model stand-in with a cheap round count."""
    return RegisteredTier1(model=_StubTier1Model(), model_id="tier1-stub", best_iteration=8)


def test_out_of_fold_scores_never_read_their_own_fold(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """Every scored row is outside the fitting set of the model that scored it.

    The structural guarantee the whole scheme rests on. Asserted over the real fold loop rather
    than over the block arithmetic alone, so that a future refactor of ``build_oof_tier1``
    cannot quietly reintroduce the contamination.
    """
    result = build_oof_tier1(
        oof_splits["train"],
        oof_splits["val"],
        source="ieee_cis",
        registered=registered_stub,
        block_count=4,
    )
    event_time = oof_splits["train"]["event_time"]
    for fold in range(2, 5):
        fit_mask = result.blocks.rows_before(fold)
        score_mask = result.blocks.rows_in(fold)
        assert not (fit_mask & score_mask).any()
        assert event_time.to_numpy()[fit_mask].max() < event_time.to_numpy()[score_mask].min()


def test_block_one_has_no_out_of_fold_score_and_is_excluded(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """Block 1 comes back NaN and is flagged unscoreable, not filled with a sentinel."""
    result = build_oof_tier1(
        oof_splits["train"],
        oof_splits["val"],
        source="ieee_cis",
        registered=registered_stub,
        block_count=4,
    )
    first_block = result.blocks.rows_in(1)
    assert np.isnan(result.scores[first_block]).all()
    assert not result.scoreable[first_block].any()

    # Everything else is scored, and scored inside the unit interval.
    rest = ~first_block
    assert not np.isnan(result.scores[rest]).any()
    assert result.scoreable[rest].all()
    assert result.scores[rest].min() >= 0.0
    assert result.scores[rest].max() <= 1.0


def test_each_fold_refits_its_own_input_spec(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """The folds carry distinct feature versions, proving the spec was not shared.

    Sharing the full-train spec across folds is the cheap mistake here: the frequency encoders
    and categorical level sets are fitted on the frame they are given, so a shared spec carries
    a held-out block's own category frequencies into its own features.
    """
    result = build_oof_tier1(
        oof_splits["train"],
        oof_splits["val"],
        source="ieee_cis",
        registered=registered_stub,
        block_count=4,
    )
    assert len(result.feature_versions) == 3
    assert len(set(result.feature_versions)) > 1, (
        "every fold produced the same feature version, which means one spec was reused "
        "across folds rather than refitted per fold"
    )


def test_fold_handicap_table_is_measured_not_asserted(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """Each fold reports its own validation PR-AUC against a common yardstick.

    The scheme's known weakness -- fold models are fitted on less data than the model that
    serves -- is reported as a number rather than argued away in prose.
    """
    result = build_oof_tier1(
        oof_splits["train"],
        oof_splits["val"],
        source="ieee_cis",
        registered=registered_stub,
        block_count=4,
    )
    assert [row.fold for row in result.handicap] == [2, 3, 4]
    # Fitting sets grow strictly with the fold number.
    rows = [row.train_rows for row in result.handicap]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    for row in result.handicap:
        assert 0.0 <= row.validation_pr_auc <= 1.0
        assert row.train_positives > 0
    assert 0.0 <= result.full_train_validation_pr_auc <= 1.0


def test_out_of_fold_rounds_come_from_the_registered_model(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """Folds use the registered best_iteration, never their own early stopping.

    Early stopping inside a fold would choose the round count by reading the very rows whose
    out-of-fold scores are being produced.
    """
    result = build_oof_tier1(
        oof_splits["train"],
        oof_splits["val"],
        source="ieee_cis",
        registered=registered_stub,
        block_count=4,
    )
    assert result.rounds == registered_stub.best_iteration


def test_out_of_fold_refuses_paysim(
    oof_splits: dict[str, pd.DataFrame], registered_stub: RegisteredTier1
) -> None:
    """The corpus guard is enforced at the point the folds are built, not only at the driver."""
    with pytest.raises(ValueError, match="simulator artefact"):
        build_oof_tier1(
            oof_splits["train"],
            oof_splits["val"],
            source="paysim",
            registered=registered_stub,
            block_count=4,
        )


# --- The model, calibration and explanations ---------------------------------------------


def _toy_model(seed: int = 0, n: int = 1_500) -> tuple[MetaModel, np.ndarray, np.ndarray]:
    """Fit a small meta-model on synthetic data, returning it with a held-out slice."""
    rng = np.random.default_rng(seed)
    names = ["tier1_score", "amount_log", "tier2_error", "tier2_is_scoreable"]
    matrix = rng.random((n, len(names)))
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] + rng.normal(0, 0.3, n)) > 1.1
    cut = int(0.7 * n)
    booster = fit_booster(
        matrix[:cut],
        labels[:cut],
        feature_names=names,
        rounds=ABLATION_NUM_ROUNDS,
        max_depth=ABLATION_MAX_DEPTH,
    )
    calibrator = fit_calibrator(booster, matrix[cut:], labels[cut:], feature_names=names)
    spec = MetaInputSpec(
        source_dataset="ieee_cis",
        blocks=("tier1", "engineered", "tier2"),
        feature_names=tuple(names),
        tier_model_versions={"tier1": "t1", "tier2": "t2"},
        upstream_feature_versions={"pipeline": "fv_synthetic"},
    )
    model = MetaModel(
        model_id="meta-learner-test",
        spec=spec,
        booster=booster,
        calibrator=calibrator,
        review_threshold=0.30,
        block_threshold=0.90,
    )
    return model, matrix[cut:], labels[cut:]


def test_sigmoid_calibration_preserves_pr_auc_exactly() -> None:
    """A strictly monotone calibrator cannot change the ranking, so PR-AUC is identical.

    This is the whole argument for shipping a sigmoid over isotonic, and asserting it here is
    what stops the argument from being merely rhetorical. Isotonic is piecewise-constant: it
    ties scores, and a calibrator that moves the headline metric is a calibrator that lies
    about the model.
    """
    model, matrix, labels = _toy_model()
    margins = model.margins(matrix)
    calibrated = model.calibrator.apply(margins)
    assert pr_auc(labels, margins) == pr_auc(labels, calibrated)


def test_calibrated_probabilities_are_monotone_in_the_margin() -> None:
    """Ordering is preserved, which is what lets SHAP in margin space explain the probability."""
    model, matrix, _ = _toy_model()
    margins = model.margins(matrix)
    calibrated = model.calibrator.apply(margins)
    ordered = calibrated[np.argsort(margins)]
    assert bool(np.all(np.diff(ordered) >= -1e-12))
    assert float(calibrated.min()) >= 0.0
    assert float(calibrated.max()) <= 1.0


def test_calibration_does_not_refit_the_booster() -> None:
    """The calibrator is fitted on held-out rows and must leave the booster untouched.

    Guards against a future edit reintroducing something that refits: the calibration rows are
    held out precisely so the booster never sees them.
    """
    model, matrix, labels = _toy_model()
    before = model.booster.trees_to_dataframe().shape
    predictions_before = model.margins(matrix).copy()
    fit_calibrator(model.booster, matrix, labels, feature_names=list(model.spec.feature_names))
    assert model.booster.trees_to_dataframe().shape == before
    assert np.array_equal(model.margins(matrix), predictions_before)


def test_shap_contributions_sum_to_the_raw_margin() -> None:
    """The real correctness test for the explanations: they must reconstruct the prediction.

    "It returned three names without crashing" is not a test of an attribution method.
    """
    model, matrix, _ = _toy_model()
    contributions = model.contributions(matrix[:20])
    margins = model.margins(matrix[:20])
    assert contributions.shape == (20, len(model.spec.feature_names) + 1)
    assert np.abs(contributions.sum(axis=1) - margins).max() < 1e-4


def test_top_features_are_ranked_by_absolute_contribution() -> None:
    """Attribution is ordered by magnitude, and every name is a real feature."""
    model, matrix, _ = _toy_model()
    top = model.explain(matrix[0], top_k=3)
    assert len(top) == 3
    assert all(name in model.spec.feature_names for name, _ in top)
    magnitudes = [abs(value) for _, value in top]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_meta_learner_does_not_import_the_shap_package() -> None:
    """Pins the reason TreeSHAP comes from XGBoost rather than from ``shap``.

    ``import shap`` raises PendingDeprecationWarning from matplotlib at import time, and this
    suite runs under ``filterwarnings = ["error"]``, so importing it anywhere would fail
    collection. XGBoost's native ``pred_contribs`` is the same algorithm, verified bit-identical.
    """
    import app.models.meta_learner  # noqa: F401

    assert "shap" not in sys.modules


# --- Persistence -------------------------------------------------------------------------


def test_meta_model_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """Reloading reproduces the scores exactly, and the artefact carries no pickle."""
    model, matrix, _ = _toy_model()
    model.save(tmp_path)
    restored = MetaModel.load(model.model_id, tmp_path)

    assert np.allclose(restored.margins(matrix), model.margins(matrix))
    assert restored.spec.feature_names == model.spec.feature_names
    assert restored.review_threshold == model.review_threshold
    # Plain JSON, so nothing in the load path can execute arbitrary code. Tier-2 and Tier-3 both
    # state this rule; a Phase 7 endpoint makes the load path reachable from a request.
    sidecar = json.loads((tmp_path / f"{model.model_id}.meta.json").read_text(encoding="utf-8"))
    assert sidecar["calibrator"] == {"a": model.calibrator.a, "b": model.calibrator.b}


@pytest.mark.parametrize("bad", ["../escape", "a/b", "C:secret", "", ".."])
def test_artifact_paths_reject_a_traversing_model_id(tmp_path: Path, bad: str) -> None:
    """Model ids are resolved through the shared guard, never concatenated into a path.

    Traversal only. ``artifact_path`` deliberately does not enforce the lowercase naming
    convention -- that is ``build_model_id``'s job -- so a merely unconventional id is allowed
    through here and only a path-escaping one is refused. ``C:secret`` is the Windows case the
    shared guard exists for: pathlib discards the base directory when a segment carries a drive.
    """
    model, _, _ = _toy_model()
    model.model_id = bad
    with pytest.raises(ValueError):
        model.save(tmp_path)


# --- The serving contract ----------------------------------------------------------------


def _transaction(**overrides: object) -> TransactionFeatures:
    """Build a scoring transaction carrying the engineered keys the toy spec needs."""
    payload: dict[str, object] = {
        "transaction_id": "t1",
        "source_dataset": "ieee_cis",
        "event_time": datetime(2018, 5, 5, tzinfo=UTC),
        "amount": Decimal("120.50"),
        "account_id": "c1",
        "feature_version": "fv_synthetic",
        "features": {"amount_log": 4.8},
    }
    payload.update(overrides)
    return TransactionFeatures(**payload)  # type: ignore[arg-type]


def _tier1(score: float = 0.7) -> Tier1Result:
    """Build a Tier-1 result."""
    return Tier1Result(score=score, is_anomaly=score > 0.5, latency_ms=1.0, model_version="t1")


def test_predict_returns_a_calibrated_explained_decision() -> None:
    """The happy path: a probability, a decision, and the top three contributions."""
    model, _, _ = _toy_model()
    result = model.predict(
        _transaction(),
        _tier1(),
        Tier2Result(
            reconstruction_error=0.4,
            is_anomaly=True,
            is_scoreable=True,
            sequence_length=12,
            latency_ms=1.0,
            model_version="t2",
        ),
        None,
    )
    assert 0.0 <= result.probability <= 1.0
    assert result.decision in {"allow", "review", "block"}
    assert len(result.top_features) == 3
    assert result.model_version == "meta-learner-test"


def test_abstention_becomes_a_sentinel_plus_an_indicator_never_a_zero() -> None:
    """An abstaining Tier-2 sets the sentinel and clears its indicator, and marks degraded.

    A 0.0 in ``tier2_error`` would read to a tree as "Tier-2 looked and found this maximally
    normal", about a layer that declined to look. Phases 3 and 4 both deferred this to here.
    """
    model, _, _ = _toy_model()
    abstained = Tier2Result(
        reconstruction_error=None,
        is_anomaly=False,
        is_scoreable=False,
        sequence_length=2,
        abstention_reason="sequence too short",
        latency_ms=1.0,
        model_version="t2",
    )
    row, reasons = model.build_vector(_transaction(), _tier1(), abstained, None)
    names = list(model.spec.feature_names)
    assert row[names.index("tier2_error")] == ABSTENTION_SENTINEL
    assert row[names.index("tier2_is_scoreable")] == 0.0
    assert reasons

    result = model.predict(_transaction(), _tier1(), abstained, None)
    assert result.degraded is True
    assert result.degraded_reason


def test_a_missing_tier_is_not_the_same_as_an_abstaining_one() -> None:
    """Both take the sentinel, but a missing Tier-1 is reported differently from an abstention."""
    model, _, _ = _toy_model()
    row, reasons = model.build_vector(_transaction(), None, None, None)
    names = list(model.spec.feature_names)
    assert row[names.index("tier1_score")] == ABSTENTION_SENTINEL
    assert any("tier1" in reason for reason in reasons)


def test_predict_raises_on_a_missing_engineered_feature() -> None:
    """A missing engineered key is an error, never a zero.

    Zero is a real value for most of these columns, so filling one would make a fabricated
    input indistinguishable from a measured one.
    """
    model, _, _ = _toy_model()
    with pytest.raises(ValueError, match="missing from the transaction"):
        model.build_vector(_transaction(features={}), _tier1(), None, None)


def test_predict_raises_on_a_feature_version_mismatch() -> None:
    """Scoring across feature definitions would silently compare different quantities."""
    model, _, _ = _toy_model()
    with pytest.raises(ValueError, match="feature_version mismatch"):
        model.build_vector(_transaction(feature_version="fv_other"), _tier1(), None, None)


def test_decisions_are_monotone_in_probability() -> None:
    """Higher risk never yields a softer action, and review sits below block."""
    model, _, _ = _toy_model()
    assert model.review_threshold <= model.block_threshold
    order = {"allow": 0, "review": 1, "block": 2}
    seen = [order[model.decide(p)] for p in np.linspace(0.0, 1.0, 60)]
    assert seen == sorted(seen)
    assert model.decide(0.0) == "allow"
    assert model.decide(1.0) == "block"


def test_top_features_fit_the_audit_record() -> None:
    """The audit trail can carry the explanation it is supposed to preserve.

    ``AuditRecord`` is ``extra="forbid"``, so this would have been unsatisfiable in Phase 7 had
    the field not been added here.
    """
    model, _, _ = _toy_model()
    result = model.predict(_transaction(), _tier1(), None, None)
    record = AuditRecord(
        transaction_id="t1",
        decided_at=datetime(2018, 5, 5, tzinfo=UTC),
        decision=result.decision,
        risk_probability=result.probability,
        feature_version="fv_synthetic",
        top_features=result.top_features,
    )
    assert record.top_features == tuple(result.top_features)


# --- The keep rule -----------------------------------------------------------------------


def _delta(block: str, kind: str, lower: float, upper: float) -> AblationDelta:
    """Build an ablation delta with a chosen interval."""
    return AblationDelta(
        block=block,
        kind=kind,
        delta=(lower + upper) / 2,
        interval=(lower, upper),
        rows=31_003,
        positives=1_131,
    )


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-0.01, 0.02, False),
        (0.001, 0.02, True),
        (0.0, 0.02, False),
        (-0.05, -0.01, False),
    ],
)
def test_keep_rule_requires_a_strictly_positive_lower_bound(
    lower: float, upper: float, expected: bool
) -> None:
    """A block is kept only when the interval excludes zero on the positive side.

    An interval touching zero exactly is a tie, and a tie does not earn a place in the serving
    path.
    """
    assert _delta("tier2", "leave_one_out", lower, upper).excludes_zero_positively is expected


def test_a_block_survives_on_either_direction() -> None:
    """Leave-one-out and add-one both count.

    Leave-one-out alone can retire two mutually redundant blocks that are jointly useful;
    add-one alone can miss a block that only helps in combination.
    """
    verdict = BlockVerdict(
        block="tier2",
        retained=True,
        exempt=False,
        leave_one_out=_delta("tier2", "leave_one_out", -0.004, 0.006),
        add_one=_delta("tier2", "add_one", 0.002, 0.011),
    )
    assert verdict.retained
    assert "retained" in verdict.rationale()


def test_a_retirement_is_worded_as_absence_of_evidence() -> None:
    """An underpowered comparison must not be reported as a measurement of no effect.

    At the arbiter slice's sample size most retirements will be ties, and the distinction
    between "we could not detect an effect" and "there is no effect" is the difference between
    an honest claim and a false one.
    """
    verdict = BlockVerdict(
        block="tier3_served",
        retained=False,
        exempt=False,
        leave_one_out=_delta("tier3_served", "leave_one_out", -0.004, 0.006),
        add_one=None,
    )
    rationale = verdict.rationale()
    assert "for want of evidence" in rationale
    assert "absence of evidence" in rationale
    assert "31,003" in rationale


def test_base_blocks_are_exempt_from_retirement() -> None:
    """Retiring tier1 would be a bug signal, not a finding, so it is not on the table."""
    verdict = BlockVerdict(
        block="tier1", retained=True, exempt=True, leave_one_out=None, add_one=None
    )
    assert "exempt" in verdict.rationale()


# --- Corpus discipline at the driver ------------------------------------------------------


def test_driver_refuses_paysim(tmp_path: Path) -> None:
    """The guard holds at the entry point, not only deep in feature assembly."""
    with pytest.raises(ValueError, match="simulator artefact"):
        load_splits(tmp_path, "paysim", None)


def test_validation_slices_are_chronological_and_disjoint() -> None:
    """V-fit precedes V-arb precedes V-late, with no row in two slices."""
    rng = np.random.default_rng(0)
    rows = 3_000
    start = datetime(2018, 4, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "event_time": [start + timedelta(minutes=5 * i) for i in range(rows)],
            "is_fraud": rng.random(rows) < 0.05,
            "amount": rng.lognormal(4, 1, rows),
        }
    )
    slices = validation_slices(frame)
    assert slices.fit["event_time"].max() < slices.arbiter["event_time"].min()
    assert slices.arbiter["event_time"].max() < slices.late["event_time"].min()
    assert len(slices.fit) + len(slices.arbiter) + len(slices.late) == rows


def test_validation_slices_refuse_a_slice_with_no_positives() -> None:
    """A slice with no positives cannot calibrate, threshold or measure a delta."""
    start = datetime(2018, 4, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "event_time": [start + timedelta(minutes=5 * i) for i in range(600)],
            "is_fraud": [False] * 600,
            "amount": [10.0] * 600,
        }
    )
    with pytest.raises(ValueError, match="no positives"):
        validation_slices(frame)


def test_block_threshold_never_collapses_onto_review() -> None:
    """Blocking must sit strictly above review, or be disabled outright.

    The regression guard for a real bug: the first implementation took the lowest threshold
    reaching the precision bar anywhere and then clamped it up to the review point. On a
    well-separated model that made the two equal, which silently removed the human-review band
    and turned every flag into a decline.
    """
    rng = np.random.default_rng(0)
    n = 4_000
    labels = rng.random(n) < 0.034
    margin = np.where(labels, rng.normal(2.0, 1.5, n), rng.normal(-6.0, 1.5, n))
    probabilities = 1.0 / (1.0 + np.exp(-5.83 * margin + 2.87))
    cost_model = CostModel.scaled_to(68.769, "IEEE-CIS amount units (consistent with USD)")

    operating = choose_operating_points(labels, probabilities, rng.lognormal(4, 1, n), cost_model)
    assert operating.block_threshold != operating.review_threshold
    assert operating.block_threshold > operating.review_threshold
    if operating.block_threshold == BLOCK_DISABLED:
        # Disabled is a legitimate outcome and must be stated, not silently produced.
        assert "disabled" in operating.block_criterion
        assert (probabilities >= operating.block_threshold).sum() == 0


def test_empirical_cdf_does_not_tie_scores() -> None:
    """The reference map must preserve ranking, not bucket it.

    The regression guard for a measured bug: applying the map by ``searchsorted`` collapsed
    Tier-1's 88,069 distinct test scores onto the grid size and cost 0.0073 PR-AUC in ties, in
    the meta-learner's single strongest feature. Interpolation keeps it lossless.
    """
    rng = np.random.default_rng(0)
    reference = rng.beta(0.5, 8.0, size=50_000)
    probe = rng.beta(0.5, 8.0, size=20_000)
    mapped = EmpiricalCdf.fit(reference).apply(probe)

    assert len(np.unique(mapped)) == len(np.unique(probe))
    assert list(np.argsort(mapped, kind="stable")) == list(np.argsort(probe, kind="stable"))


def test_a_null_engineered_feature_becomes_nan_not_an_error() -> None:
    """A null value is a measurement; only an absent key is an error.

    ``seconds_since_prior_txn`` is genuinely null on an account's first transaction. The
    training matrix carried such values through as NaN and the booster learned a default
    direction for them, so serving must present NaN rather than refusing the row or imputing a
    number that would fabricate a history the account does not have.
    """
    model, _, _ = _toy_model()
    row, _ = model.build_vector(_transaction(features={"amount_log": None}), _tier1(), None, None)
    assert np.isnan(row[list(model.spec.feature_names).index("amount_log")])


def test_an_absent_engineered_key_still_raises() -> None:
    """The paired half: a missing key means a definition mismatch and must not be filled."""
    model, _, _ = _toy_model()
    with pytest.raises(ValueError, match="missing from the transaction"):
        model.build_vector(_transaction(features={}), _tier1(), None, None)


def test_public_projection_withholds_the_attribution() -> None:
    """The response-safe view must not carry feature attribution.

    Returning attribution to whoever submitted the transaction tells them which signals to
    avoid next time. The security checklist treats that as an evasion oracle and the track
    treats it as a disqualification, so omitting it is something a caller has to undo rather
    than something they have to remember.
    """
    model, _, _ = _toy_model()
    result = model.predict(_transaction(), _tier1())
    assert result.top_features

    public = result.public()
    assert "top_features" not in public
    assert set(public) == {"probability", "decision", "model_version"}


def test_attribution_can_be_skipped_on_the_hot_path() -> None:
    """TreeSHAP is a second full pass over the ensemble; a caller that discards it may opt out."""
    model, _, _ = _toy_model()
    assert model.predict(_transaction(), _tier1(), explain=False).top_features == ()


def test_attribution_is_immutable() -> None:
    """pydantic's frozen=True is shallow, so the attribution must not be a mutable list.

    It ends up on an audit record whose entire purpose is tamper-evidence.
    """
    model, _, _ = _toy_model()
    result = model.predict(_transaction(), _tier1())
    assert isinstance(result.top_features, tuple)
    with pytest.raises((AttributeError, TypeError)):
        result.top_features.append(("x", 1.0))  # type: ignore[attr-defined]


def test_deny_list_sees_through_a_tier3_prefix() -> None:
    """A prefixed label read must not slip past the guard.

    Tier-3 columns reach the matrix prefixed, but the deny list holds bare names. A naive set
    intersection waves ``tier3_account_is_fraudulent`` through -- a direct label read matched
    against a guard that cannot see it. The allowlist in ``TIER3_CARRIED_COLUMNS`` is the
    primary defence; this is the backstop, and a backstop that cannot fire is not one.
    """
    for name in ("tier3_account_is_fraudulent", "tier3nc_account_is_fraudulent"):
        assert meta_denied_columns_present([name]) == [name]
        with pytest.raises(ValueError, match="account_is_fraudulent"):
            require_clean_feature_names(["amount_log", name])

    # And it still passes clean prefixed columns.
    assert meta_denied_columns_present(["tier3_ring_size", "tier3nc_account_degree"]) == []


def test_a_negatively_excluding_interval_is_reported_as_harm_not_as_no_evidence() -> None:
    """An interval entirely below zero is evidence of harm, and must not be called a tie.

    The regression guard for a real bug: the retirement sentence was hardcoded to say "the
    interval does not exclude zero" for every non-retained block. On the shipped run
    ``tier3_topology`` came back at [-0.0100, -0.0026], so the report and the registry both
    asserted a falsehood about their own numbers -- and under-claimed, since "measurably
    degrades the model" is the stronger finding.
    """
    verdict = BlockVerdict(
        block="tier3_topology",
        retained=False,
        exempt=False,
        leave_one_out=_delta("tier3_topology", "leave_one_out", -0.0100, -0.0026),
        add_one=_delta("tier3_topology", "add_one", -0.0103, 0.0010),
    )
    rationale = verdict.rationale()
    assert "NEGATIVE" in rationale
    assert "degrades the model" in rationale
    assert "absence of evidence" not in rationale


def test_calibrator_is_fitted_on_the_scale_it_is_applied_to() -> None:
    """Platt must be fitted on margins, because margins are what scoring feeds it.

    The regression guard for a real shipped bug. ``CalibratedClassifierCV`` prefers
    ``decision_function`` and falls back to ``predict_proba``; with only the latter exposed it
    fitted ``(a, b)`` against values in [0, 1] while ``score_frame`` applied them to log-odds.
    The sigmoid was then evaluated far outside its fitted range and every probability collapsed
    toward zero — the shipped model had ``a = -97.5``, and ordinary clean rows scored 1e-30.

    Two assertions, because either alone is weak. The slope check catches the scale mismatch
    directly; the mean-prediction check catches whether the output is actually calibrated.
    """
    rng = np.random.default_rng(0)
    rows, features = 4_000, 4
    matrix = rng.random((rows, features))
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] + rng.normal(0, 0.3, rows)) > 1.1
    names = [f"f{index}" for index in range(features)]

    booster = fit_booster(
        matrix[:3000], labels[:3000], feature_names=names, rounds=200, max_depth=3
    )
    calibrator = fit_calibrator(booster, matrix[3000:], labels[3000:], feature_names=names)

    # A margin-scale Platt fit has a slope of order 1. Fitting across a [0, 1] probability band
    # instead produces a slope an order of magnitude or more steeper.
    assert abs(calibrator.a) < 10.0, (
        f"slope {calibrator.a:.2f} is too steep for a margin-scale fit, which means the "
        "calibrator was fitted on probabilities and is being applied to log-odds"
    )

    import xgboost as xgb

    margins = np.asarray(
        booster.predict(xgb.DMatrix(matrix[3000:], feature_names=names), output_margin=True),
        dtype="float64",
    )
    predicted = calibrator.apply(margins)
    assert abs(float(predicted.mean()) - float(labels[3000:].mean())) < 0.02


def test_scoring_honours_the_early_stopping_selection() -> None:
    """A model early-stopped at iteration k must be scored with k+1 trees, not all of them.

    The regression guard for the second shipped bug: ``Booster.predict`` uses the whole
    ensemble unless given an ``iteration_range``, so the model was scored 100 rounds past its
    own validation-selected optimum, and a reloaded artefact that honoured ``best_iteration``
    would not have reproduced the registered metrics.
    """
    model, matrix, _ = _toy_model()
    full = model.margins(matrix).copy()

    model.best_iteration = 1
    assert model._iteration_range == (0, 2)
    truncated = model.margins(matrix)
    assert not np.allclose(truncated, full), "iteration_range had no effect on scoring"

    # And it survives the artefact round trip, so a reload reproduces the registered numbers.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        model.save(Path(directory))
        restored = MetaModel.load(model.model_id, Path(directory))
    assert restored.best_iteration == 1
    assert np.allclose(restored.margins(matrix), truncated)


def test_dropped_columns_round_trip_through_the_string_form() -> None:
    """A reconstructed spec must hash to the same feature_version as the fitted one.

    The regression guard for a provenance bug. ``load_registered_tier1`` rebuilt Tier-1's spec
    with ``dropped=()``, but ``dropped`` is hashed into the feature version, so the meta-learner
    recorded an upstream ``fv_`` that resolved to nothing in the registry — a traceability chain
    that looked intact and was not, which is worse than recording nothing at all.
    """
    from app.models.meta_features import _parse_dropped
    from app.models.tier1_features import DroppedColumn

    original = (
        DroppedColumn(column="velocity_available", reason="constant on train"),
        DroppedColumn(column="odd_one", reason="null rate above 0.999"),
    )
    serialised = [str(item) for item in original]
    rebuilt = _parse_dropped(serialised)

    assert [str(item) for item in rebuilt] == serialised
    assert [(d.column, d.reason) for d in rebuilt] == [(d.column, d.reason) for d in original]
