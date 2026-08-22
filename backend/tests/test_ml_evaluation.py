"""Tests for the shared evaluation, cost and registry machinery.

Separate from ``test_tier1.py`` because these modules are not Tier-1's: Phases 3 through 6
each report through them, and the guarantees asserted here are what stop a later tier from
quietly reporting to a lower standard than an earlier one.

The bar being enforced is ``.claude/skills/ml-evaluation-standards/SKILL.md``:

- a PR-AUC is never reported without its no-skill floor (section 2);
- a result is never reported without a false-positive cost estimate carrying its assumptions
  in plain language (section 3);
- a suspiciously high PR-AUC is flagged as a probable leak rather than celebrated (section 4);
- the registry is append-only, so a past prediction stays traceable (section 5).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.ml.cost import (
    DEFAULT_REVIEW_COST,
    IEEE_CIS_MEDIAN_AMOUNT,
    CostModel,
    choose_threshold_by_cost,
    cost_at_threshold,
    render_sensitivity,
    review_cost_sweep,
    sensitivity_sweep,
    threshold_for_flag_rate,
)
from app.ml.evaluation import (
    LEAK_SUSPICION_PR_AUC,
    bootstrap_pr_auc,
    bootstrap_pr_auc_delta,
    confusion_at_threshold,
    evaluate,
    no_skill_pr_auc,
    pr_auc,
    threshold_for_recall,
)
from app.ml.registry import (
    RegistryEntry,
    append_entry,
    build_model_id,
    find_entry,
    latest_entry,
    read_registry,
)


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(labels, scores, amounts)`` for a separable-but-imperfect toy problem."""
    rng = np.random.default_rng(0)
    labels = rng.random(2_000) < 0.08
    scores = np.clip(rng.random(2_000) + 0.35 * labels, 0.0, 1.0)
    amounts = rng.lognormal(4.0, 1.0, 2_000)
    return labels, scores, amounts


# --- Metrics -----------------------------------------------------------------------


def test_confusion_matrix_derives_rates_from_raw_counts() -> None:
    """Precision, recall and F1 follow from TN/FP/FN/TP."""
    labels = np.array([True, True, False, False, False])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    matrix = confusion_at_threshold(labels, scores, 0.5)
    assert matrix.to_dict() == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert matrix.precision == pytest.approx(0.5)
    assert matrix.recall == pytest.approx(0.5)
    assert matrix.f1 == pytest.approx(0.5)
    assert matrix.total == 5


def test_confusion_matrix_survives_flagging_nothing() -> None:
    """Precision is 0.0 rather than a division by zero when nothing is flagged."""
    labels = np.array([True, False, False])
    matrix = confusion_at_threshold(labels, np.array([0.1, 0.1, 0.1]), 0.9)
    assert matrix.to_dict() == {"tn": 2, "fp": 0, "fn": 1, "tp": 0}
    assert matrix.precision == 0.0
    assert matrix.f1 == 0.0


def test_no_skill_floor_equals_the_base_rate() -> None:
    """A random ranker's average precision is the positive base rate.

    The property the whole reporting format rests on: it is what makes a PR-AUC of 0.28
    excellent on one corpus and near-worthless on another.
    """
    rng = np.random.default_rng(3)
    labels = rng.random(20_000) < 0.03
    assert pr_auc(labels, rng.random(20_000)) == pytest.approx(no_skill_pr_auc(labels), abs=0.01)


def test_pr_auc_is_zero_when_there_are_no_positives() -> None:
    """A split with no positives yields 0.0 rather than a NaN that propagates into a report."""
    labels = np.zeros(50, dtype=bool)
    assert pr_auc(labels, np.random.default_rng(0).random(50)) == 0.0
    assert bootstrap_pr_auc(labels, np.zeros(50)) == (0.0, 0.0)


def test_bootstrap_interval_brackets_the_point_estimate(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The resampled interval contains the value measured on the split itself."""
    labels, scores, _ = scored
    point = pr_auc(labels, scores)
    low, high = bootstrap_pr_auc(labels, scores, resamples=200, seed=1)
    assert low <= point <= high
    assert low < high


def test_bootstrap_delta_is_exactly_zero_for_identical_models(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Pairing on the same resample means a model compared to itself has zero spread.

    This is why the delta is bootstrapped paired rather than by differencing two independent
    intervals: the split's own variance cancels, leaving only the difference between models.
    """
    labels, scores, _ = scored
    low, high = bootstrap_pr_auc_delta(labels, scores, scores, resamples=100, seed=1)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_threshold_for_recall_reaches_the_target(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The recall-targeted threshold achieves at least the recall asked for."""
    labels, scores, _ = scored
    threshold = threshold_for_recall(labels, scores, 0.8)
    assert confusion_at_threshold(labels, scores, threshold).recall >= 0.8


# --- Honesty rules -----------------------------------------------------------------


def test_result_reports_the_floor_alongside_the_headline(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A rendered result always carries its base rate, threshold and lift."""
    labels, scores, amounts = scored
    result = evaluate(
        "toy",
        "test",
        labels,
        scores,
        threshold=0.6,
        threshold_criterion="minimising estimated cost on validation",
        cost=cost_at_threshold(labels, scores, amounts, 0.6, CostModel()),
    )
    rendered = result.render()
    assert "no-skill floor" in rendered
    assert "base rate" in rendered
    assert "Threshold:" in rendered
    assert "Confusion matrix" in rendered
    assert result.lift_over_no_skill > 1.0


def test_result_flags_a_suspicious_pr_auc_rather_than_celebrating_it() -> None:
    """A near-perfect PR-AUC on fraud data is marked as a suspected leak."""
    labels = np.array([True] * 50 + [False] * 950)
    perfect = np.concatenate([np.ones(50), np.zeros(950)])
    result = evaluate("perfect", "test", labels, perfect, threshold=0.5, threshold_criterion="test")
    assert result.pr_auc > LEAK_SUSPICION_PR_AUC
    assert result.is_leak_suspicious
    assert "suspected leak" in result.render()


def test_cost_estimate_always_states_its_assumptions(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Section 3: a cost figure without its assumptions is not reportable."""
    labels, scores, amounts = scored
    estimate = cost_at_threshold(labels, scores, amounts, 0.6, CostModel())
    rendered = estimate.render()
    assert "Assumptions:" in rendered
    assert "review capacity is unbounded" in rendered.lower()
    assert len(estimate.to_dict()["assumptions"]) >= 4
    assert "flag rate" in rendered


# --- Cost ---------------------------------------------------------------------------


def test_threshold_search_matches_brute_force(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The cumulative-sum sweep finds the same minimum as evaluating every threshold.

    The sweep exists because the naive version is quadratic in the split size. It is only
    worth having if it is exact, so it is checked against the slow version it replaced.
    """
    labels, scores, amounts = scored
    model = CostModel()
    _, estimate = choose_threshold_by_cost(labels, scores, amounts, model)
    brute = min(
        cost_at_threshold(labels, scores, amounts, float(t), model).total_cost for t in scores
    )
    assert estimate.total_cost <= brute + 1e-9


def test_threshold_search_offers_only_achievable_operating_points() -> None:
    """With tied scores, the chosen threshold is one a ``>=`` rule can actually realise."""
    labels = np.array([True, False, True, False])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    amounts = np.array([10.0, 10.0, 10.0, 10.0])
    threshold, estimate = choose_threshold_by_cost(labels, scores, amounts, CostModel())
    assert np.isinf(threshold) or threshold == 0.5
    assert estimate.flagged in (0, 4)


def test_flag_rate_ceiling_is_respected() -> None:
    """The capacity threshold keeps the realised flag rate under its ceiling."""
    rng = np.random.default_rng(1)
    scores = rng.random(10_000)
    labels = rng.random(10_000) < 0.05
    amounts = np.full(10_000, 50.0)
    estimate = cost_at_threshold(
        labels, scores, amounts, threshold_for_flag_rate(scores, 0.01), CostModel()
    )
    assert 0.005 < estimate.flag_rate <= 0.01


def test_scaled_cost_model_preserves_the_economics() -> None:
    """Rescaling to another corpus keeps the review-cost-to-typical-loss ratio fixed.

    The only thing that transfers between corpora whose amounts are on different scales. The
    absolute figures do not, which is why they carry a units label.
    """
    scaled = CostModel.scaled_to(IEEE_CIS_MEDIAN_AMOUNT * 2_500, "simulator units")
    assert scaled.review_cost / DEFAULT_REVIEW_COST == pytest.approx(2_500, rel=1e-6)
    assert scaled.review_cost / scaled.chargeback_fee == pytest.approx(
        CostModel().review_cost / CostModel().chargeback_fee
    )
    assert "simulator units" in scaled.units


def test_review_cost_sweep_moves_the_threshold_more_than_the_magnitude_sweep(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Varying the FP:FN ratio changes the recommendation; scaling both barely does.

    The reason both sweeps are reported. Section 3 asks for +/-50%, but that direction leaves
    the ratio fixed and so tends to produce a flat table that understates how much the
    recommendation rests on a guessed constant.
    """
    labels, scores, amounts = scored
    model = CostModel()
    magnitude = sensitivity_sweep(labels, scores, amounts, model)
    ratio = review_cost_sweep(labels, scores, amounts, model)

    magnitude_spread = max(r.estimate.flag_rate for r in magnitude) - min(
        r.estimate.flag_rate for r in magnitude
    )
    ratio_spread = max(r.estimate.flag_rate for r in ratio) - min(
        r.estimate.flag_rate for r in ratio
    )
    assert ratio_spread > magnitude_spread
    assert "flag rate" in render_sensitivity(ratio, "title")


# --- Registry -------------------------------------------------------------------------


def _entry(model_id: str, source: str = "ieee_cis") -> RegistryEntry:
    """Return a minimal registry entry."""
    return RegistryEntry(
        model_id=model_id,
        layer="tier1_anomaly",
        algorithm="lightgbm",
        source_dataset=source,
        feature_version="fv_test",
        training_window={"start": "2018-01-01", "end": "2018-03-01"},
        hyperparameters={"num_leaves": 63},
        random_seed=42,
        heldout_test={"pr_auc": 0.5},
    )


def test_appending_preserves_every_prior_entry(tmp_path: Path) -> None:
    """The property the registry exists for: history is never lost."""
    path = tmp_path / "registry.json"
    append_entry(_entry("tier1-a"), path)
    append_entry(_entry("tier1-b"), path)
    append_entry(_entry("tier1-c"), path)
    assert [entry["model_id"] for entry in read_registry(path)] == [
        "tier1-a",
        "tier1-b",
        "tier1-c",
    ]


def test_duplicate_model_id_is_refused(tmp_path: Path) -> None:
    """Two models may not answer to one version, or an audit row becomes ambiguous."""
    path = tmp_path / "registry.json"
    append_entry(_entry("tier1-a"), path)
    with pytest.raises(ValueError, match="append-only"):
        append_entry(_entry("tier1-a"), path)


def test_a_registry_of_unexpected_shape_is_not_overwritten(tmp_path: Path) -> None:
    """Refusing beats appending to something that would be destroyed by the write."""
    path = tmp_path / "registry.json"
    path.write_text('{"model_id": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        read_registry(path)


def test_missing_or_empty_registry_reads_as_empty(tmp_path: Path) -> None:
    """A cold clone has no registry yet; that is not an error."""
    assert read_registry(tmp_path / "absent.json") == []
    empty = tmp_path / "empty.json"
    empty.write_text("  \n", encoding="utf-8")
    assert read_registry(empty) == []


def test_lookup_by_id_and_by_layer(tmp_path: Path) -> None:
    """Entries resolve by id, and the latest per layer/corpus is the last appended."""
    path = tmp_path / "registry.json"
    append_entry(_entry("tier1-a"), path)
    append_entry(_entry("tier1-b"), path)
    append_entry(_entry("paysim-a", source="paysim"), path)

    assert find_entry("tier1-b", path) is not None
    assert find_entry("absent", path) is None
    newest = latest_entry("tier1_anomaly", "ieee_cis", path)
    assert newest is not None and newest["model_id"] == "tier1-b"
    paysim = latest_entry("tier1_anomaly", "paysim", path)
    assert paysim is not None and paysim["model_id"] == "paysim-a"
    assert latest_entry("tier3_graph", "ieee_cis", path) is None


def test_registry_file_is_valid_json_after_append(tmp_path: Path) -> None:
    """The file stays parseable — it is read by tooling, not only by this module."""
    path = tmp_path / "registry.json"
    append_entry(_entry("tier1-a"), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[0]["random_seed"] == 42


def test_model_id_is_filename_safe() -> None:
    """Ids become filename components, so unsafe characters are refused at construction.

    The whole slug is lowercased, timestamp separators included — Windows filesystems are
    case-insensitive, so two ids differing only in case would collide as artefact filenames
    while looking distinct in the registry.
    """
    from datetime import UTC, datetime

    moment = datetime(2026, 8, 22, 19, 12, 7, tzinfo=UTC)
    assert build_model_id("tier1_anomaly", "lightgbm", "ieee_cis", moment) == (
        "tier1-anomaly-lightgbm-ieee-cis-20260822t191207z"
    )
    with pytest.raises(ValueError, match="unsafe"):
        build_model_id("tier1/anomaly", "lightgbm", "ieee-cis", moment)
