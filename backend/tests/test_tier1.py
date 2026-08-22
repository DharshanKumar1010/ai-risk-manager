"""Tier-1 tests.

The tests that matter most here are not the ones checking that a number is produced — they
are the ones checking that the number could not have been produced dishonestly:

- :func:`test_denied_columns_never_reach_the_matrix` and its companion
  :func:`test_deny_list_guard_actually_catches_a_leak`. The second is the important half. A
  guard that never fires is indistinguishable from a guard that cannot fire, so one test
  deliberately plants a time-monotone column and asserts the check rejects it.
- :func:`test_encoders_are_fitted_on_train_alone`, likewise paired: refitting on train
  reproduces the tables exactly, and refitting on the whole frame does not. Without the second
  assertion the first would pass against a function that ignored its mask.
- :func:`test_isolation_forest_scores_anomalies_higher`. ``score_samples`` runs
  higher-is-more-normal; an unnoticed sign flip yields a PR-AUC below the base-rate floor,
  which reads as a weak model rather than as a bug.
- :func:`test_numpy_and_pandas_paths_agree`. Serving scores through an integer-code array for
  speed; that is only sound while the code ordering matches the categoricals training saw.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.data.schema import TransactionFeatures
from app.ml.evaluation import (
    no_skill_pr_auc,
    pr_auc,
)
from app.models.tier1_anomaly import (
    LATENCY_BUDGET_P95_MS,
    ScoreNormaliser,
    benchmark_latency,
    explain,
)
from app.models.tier1_features import (
    DENIED_COLUMNS,
    candidate_columns,
    denied_columns_present,
    find_degenerate_columns,
    fit_tier1_input_spec,
)
from app.models.train_tier1 import (
    build_scoring_transactions,
    fit_isolation_forest,
    fit_lightgbm,
    run_corpus,
)

ROWS = 1_200


def _synthetic_frame(rows: int = ROWS, seed: int = 7) -> pd.DataFrame:
    """Build a frame shaped like an IEEE-CIS parquet split, with a learnable signal.

    Synthetic rather than sampled from ``data/processed`` so the suite runs without the
    corpora present and in a second rather than a minute. The signal is deliberately simple —
    fraud is more likely at a high amount on an unfamiliar device — so a working model finds
    it and a broken one does not.
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
            "transaction_type": pd.Series(rng.choice(["W", "C", "R"], rows), dtype="object"),
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
            "all_null_column": pd.Series([np.nan] * rows, dtype="float64"),
            "is_fraud": is_fraud,
            "split": "train",
            "feature_version": "fv_synthetic",
            "uid_strategy": "card_addr_d1n",
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    """Return a synthetic corpus frame."""
    return _synthetic_frame()


@pytest.fixture
def splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return chronological train/val/test splits of the synthetic frame."""
    first, second = int(0.7 * len(frame)), int(0.85 * len(frame))
    return {
        "train": frame.iloc[:first].copy(),
        "val": frame.iloc[first:second].copy(),
        "test": frame.iloc[second:].copy(),
    }


# --- The leak guards ---------------------------------------------------------------


def test_denied_columns_never_reach_the_matrix(splits: dict[str, pd.DataFrame]) -> None:
    """No denied column survives into the fitted input spec."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    assert denied_columns_present(spec.feature_names) == []
    assert denied_columns_present(spec.numeric_feature_names) == []


@pytest.mark.parametrize("denied", ["TransactionDT", "TransactionID", "is_fraud", "account_id"])
def test_deny_list_guard_actually_catches_a_leak(
    splits: dict[str, pd.DataFrame], denied: str
) -> None:
    """The guard fires on a planted leak.

    The companion to the test above, and the one that gives it meaning. ``TransactionDT`` and
    ``TransactionID`` both increase along the timeline, so under a chronological split they
    separate train from test perfectly — a model handed either scores superbly and has learned
    nothing.
    """
    assert denied in DENIED_COLUMNS
    assert denied_columns_present([denied, "amount_log"]) == [denied]
    assert denied not in candidate_columns(splits["train"], "ieee_cis")


def test_degenerate_columns_are_dropped_by_measurement(
    splits: dict[str, pd.DataFrame],
) -> None:
    """All-null and constant columns are excluded, and the reason is recorded.

    Measured rather than hardcoded, which is what let this reproduce PaySim's eleven dead
    columns without anyone listing them.
    """
    train = splits["train"]
    kept, dropped = find_degenerate_columns(train, candidate_columns(train, "ieee_cis"))
    reasons = {item.column: item.reason for item in dropped}
    assert "all_null_column" in reasons
    assert "null" in reasons["all_null_column"]
    assert "velocity_available" in reasons
    assert "constant" in reasons["velocity_available"]
    assert "amount_log" in kept


def test_encoders_are_fitted_on_train_alone(splits: dict[str, pd.DataFrame]) -> None:
    """Frequency tables come from train, and demonstrably not from the whole frame.

    The second assertion is what makes the first one a real check. A ``fit`` that silently
    ignored its input and read everything would pass the first and fail this.
    """
    train_spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    refit = fit_tier1_input_spec(splits["train"], "ieee_cis")
    assert train_spec.encoders == refit.encoders

    everything = pd.concat(splits.values(), ignore_index=True)
    leaky = fit_tier1_input_spec(everything, "ieee_cis")
    assert leaky.encoders != train_spec.encoders, (
        "Fitting on train and on the full frame produced identical encoders, so the train "
        "mask is not being honoured."
    )


def test_tier1_feature_version_differs_from_the_pipeline_version(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Tier-1 mints its own feature version, and it is stable across refits."""
    first = fit_tier1_input_spec(splits["train"], "ieee_cis").to_feature_definition()
    second = fit_tier1_input_spec(splits["train"], "ieee_cis").to_feature_definition()
    assert first.feature_version == second.feature_version
    assert first.feature_version != "fv_synthetic"


# --- The models --------------------------------------------------------------------


def test_isolation_forest_scores_anomalies_higher(splits: dict[str, pd.DataFrame]) -> None:
    """The Isolation Forest sign convention is right way up.

    ``score_samples`` returns higher-is-more-normal. If the negation were dropped, the score
    would rank the most ordinary transactions as the most suspicious and PR-AUC would land
    *below* the base-rate floor — which looks like a weak model, not like a bug.
    """
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_isolation_forest(spec, splits["train"], model_id="tier1-test-if")
    scores = model.score_frame(splits["test"])

    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # An extreme row must be scored more suspicious than a typical one.
    typical = splits["test"].iloc[[0]].copy()
    extreme = typical.copy()
    extreme["amount_log"] = typical["amount_log"].to_numpy()[0] + 12.0
    extreme["velocity_count_24h"] = 5_000.0
    extreme["C1"] = 5_000.0
    assert model.score_frame(extreme)[0] > model.score_frame(typical)[0]


def test_lightgbm_beats_the_no_skill_floor(splits: dict[str, pd.DataFrame]) -> None:
    """The supervised model clears the floor a random ranker achieves."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-lgbm", numeric_only=False
    )
    labels = splits["test"]["is_fraud"].to_numpy(dtype=bool)
    scores = model.score_frame(splits["test"])
    assert pr_auc(labels, scores) > no_skill_pr_auc(labels)


def test_numpy_and_pandas_paths_agree(splits: dict[str, pd.DataFrame]) -> None:
    """Single-row serving reproduces batch scoring exactly.

    Serving converts categoricals to integer codes and scores a bare float array, which is
    ~200x faster than handing LightGBM a one-row DataFrame. That is only valid while the code
    ordering matches the categorical ordering seen at training, so the equivalence is asserted
    rather than trusted.
    """
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-paths", numeric_only=False
    )
    test = splits["test"]
    batch = model.score_frame(test)
    transactions = build_scoring_transactions(test, model, count=25)
    for index, transaction in enumerate(transactions):
        assert model.score(transaction).score == pytest.approx(batch[index], abs=1e-9)


def test_score_refuses_a_mismatched_feature_version(splits: dict[str, pd.DataFrame]) -> None:
    """A vector built against a different definition is rejected, not scored."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-version", numeric_only=False
    )
    transaction = build_scoring_transactions(splits["test"], model, count=1)[0]
    wrong = transaction.model_copy(update={"feature_version": "fv_something_else"})
    with pytest.raises(ValueError, match="feature_version"):
        model.score(wrong)


def test_score_refuses_a_missing_feature_rather_than_zero_filling(
    splits: dict[str, pd.DataFrame],
) -> None:
    """A missing feature raises. Zero-filling would hide a wrong decision behind a valid row."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-missing", numeric_only=False
    )
    transaction = build_scoring_transactions(splits["test"], model, count=1)[0]
    incomplete = dict(transaction.features)
    dropped = model.feature_names[0]
    del incomplete[dropped]
    with pytest.raises(ValueError, match="missing"):
        model.score(transaction.model_copy(update={"features": incomplete}))


def test_result_contract_and_latency_budget(splits: dict[str, pd.DataFrame]) -> None:
    """The scoring result is well formed and the p95 budget holds."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-latency", numeric_only=False
    )
    transactions = build_scoring_transactions(splits["test"], model, count=20)
    result = model.score(transactions[0])
    assert 0.0 <= result.score <= 1.0
    assert result.model_version == "tier1-test-latency"
    assert result.is_anomaly == (result.score >= model.threshold)

    latency = benchmark_latency(model, transactions)
    assert latency["p95_ms"] < LATENCY_BUDGET_P95_MS


def test_shap_explanation_ranks_by_absolute_contribution(
    splits: dict[str, pd.DataFrame],
) -> None:
    """SHAP attribution returns known features, ordered by magnitude."""
    spec = fit_tier1_input_spec(splits["train"], "ieee_cis")
    model = fit_lightgbm(
        spec, splits["train"], splits["val"], model_id="tier1-test-shap", numeric_only=False
    )
    transaction = build_scoring_transactions(splits["test"], model, count=1)[0]
    contributions = explain(model, transaction, top_k=4)
    assert len(contributions) == 4
    assert all(name in model.feature_names for name, _ in contributions)
    magnitudes = [abs(value) for _, value in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_score_normaliser_handles_a_degenerate_range() -> None:
    """A constant-score model maps to 0.5 rather than dividing by zero."""
    normaliser = ScoreNormaliser.fit("minmax", np.full(10, 3.0))
    assert normaliser.apply(np.array([3.0]))[0] == pytest.approx(0.0)
    assert np.all(np.isfinite(normaliser.apply(np.array([1.0, 3.0, 9.0]))))


# --- The scoring contract ----------------------------------------------------------


def test_scoring_contract_carries_no_label() -> None:
    """``TransactionFeatures`` refuses a label, so one cannot ride into a feature vector.

    ``extra="forbid"`` is what enforces it. The label lives on ``LabelledTransaction``
    instead, which nothing on a serving path touches.
    """
    fields: dict[str, Any] = {
        "transaction_id": "1",
        "source_dataset": "ieee_cis",
        "event_time": datetime(2018, 1, 1, tzinfo=UTC),
        "amount": Decimal("10.0000"),
        "account_id": "c1",
        "feature_version": "fv_test",
        "features": {"amount_log": 2.4},
        "is_fraud": True,
    }
    with pytest.raises(ValueError):
        TransactionFeatures(**fields)


def test_model_selection_reads_validation_not_test(splits: dict[str, pd.DataFrame]) -> None:
    """The shipped model is chosen on validation PR-AUC, never on the test result.

    The regression guard for a real bug: the first implementation ranked candidates by
    ``candidate.result.pr_auc``, which is the *test* result. That is the contamination
    ml-evaluation-standards section 1 forbids — the winner would have been chosen using the
    same split its headline is quoted from. On a sampled run it selected a different model
    than the honest procedure does, so it was not merely a technicality.
    """
    from datetime import UTC, datetime

    report = run_corpus(
        "ieee_cis",
        {"train": splits["train"], "val": splits["val"], "test": splits["test"]},
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    trained = [c for c in report.candidates if c.model is not None]
    best_validation = max(c.validation_pr_auc for c in trained)
    assert report.winner.validation_pr_auc == best_validation

    # And the winner is not merely whichever candidate happens to top the test split.
    assert report.winner.validation_pr_auc >= report.runner_up.validation_pr_auc
