"""Tests for the Phase 6 causal cost layer.

This phase's whole claim is that ranking by expected cost beats ranking by probability, and the
things that could make that claim false are not subtle modelling errors -- they are a leaked
feature, a fold that scores rows it was fitted on, a propensity that cannot be inverted, or an
arithmetic identity that quietly stopped holding. So the tests here are weighted towards the
guards, and **every guard is paired with a test that plants a violation and asserts it fires**.
A guard that never fires is indistinguishable from a guard that cannot fire.

The other weight is on the algebra. The module docstring in ``app.models.causal_cost`` proves
that the DR-Learner collapses onto a cost-weighted plug-in on this data. A proof in a docstring
that nothing checks is a comment, so the identities it rests on are asserted here.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.cost import (
    CostModel,
    cnp_cost_model,
    cost_curve,
    threshold_for_flag_rate,
    value_recall_at_flag_rate,
    value_recall_at_threshold,
)
from app.ml.evaluation import (
    bootstrap_cost_delta,
    bootstrap_value_recall_delta,
    confusion_at_threshold,
    evaluate,
    format_threshold,
)
from app.ml.ope import (
    POSITIVITY_FLOOR,
    LoggingPolicy,
    assert_positivity,
    direct_method,
    doubly_robust,
    evaluate_policy,
    ipw,
    realised_cost,
    simulate_logging_policy,
    true_policy_cost,
)
from app.models.causal_cost import (
    CostPolicy,
    DecisionCost,
    LossModel,
    amount_aware_threshold,
    expected_cost_if_allowed,
    expected_cost_if_blocked,
    fit_loss_model,
    loss_target,
    out_of_fold_loss,
)
from app.models.meta_features import assert_forward_chaining, time_blocks
from app.models.train_cost_learner import scores_under

ROWS = 1_200
SEED = 7


def _synthetic_frame(rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build a frame shaped like an IEEE-CIS parquet split, with a planted cost structure.

    Synthetic rather than sampled from ``data/processed`` so the suite runs without the corpora
    present, and the same shape ``test_tier1`` and ``test_meta_learner`` use.

    The planted signal is deliberately *different* from those files: risk rises with a new
    device, but the large amounts sit on a partly disjoint set of rows. That is the structure
    this phase exists to exploit -- if amount and risk were perfectly aligned, cost-aware
    ranking and probability ranking would agree and nothing here would be measurable.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2018, 1, 1, tzinfo=UTC)
    device_is_new = rng.random(rows) < 0.4
    big_ticket = rng.random(rows) < 0.15
    amount = np.where(
        big_ticket,
        rng.lognormal(mean=7.0, sigma=0.6, size=rows),
        rng.lognormal(mean=3.5, sigma=0.8, size=rows),
    )
    risk = 0.02 + 0.30 * device_is_new + 0.05 * big_ticket
    is_fraud = rng.random(rows) < risk
    return pd.DataFrame(
        {
            "transaction_id": [f"t{index}" for index in range(rows)],
            "source_dataset": "ieee_cis",
            "event_time": [start + timedelta(minutes=3 * index) for index in range(rows)],
            "amount": amount,
            "amount_log": np.log1p(amount),
            "account_id": [f"c{index % 200}" for index in range(rows)],
            "transaction_type": rng.choice(["W", "C", "R"], size=rows),
            "is_fraud": is_fraud,
            "device_is_new": device_is_new,
            "addr_is_new": rng.random(rows) < 0.3,
            "device_mismatch": rng.random(rows),
            "addr_mismatch": rng.random(rows),
            "hour_of_day": rng.integers(0, 24, size=rows),
            "day_of_week": rng.integers(0, 7, size=rows),
            "account_prior_txn_count": rng.integers(0, 30, size=rows),
            "velocity_count_1h": rng.random(rows) * 3,
            "velocity_sum_1h": rng.random(rows) * 300,
            "split": "train",
            "feature_version": "fv_synthetic",
        }
    )


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the numeric feature matrix the loss regression consumes."""
    return frame.loc[:, list(FEATURES)].to_numpy(dtype="float64")


FEATURES = (
    "amount_log",
    "device_is_new",
    "addr_is_new",
    "device_mismatch",
    "addr_mismatch",
    "hour_of_day",
    "day_of_week",
    "account_prior_txn_count",
    "velocity_count_1h",
    "velocity_sum_1h",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Share one synthetic frame: several tests fit a booster on it."""
    return _synthetic_frame()


# ==========================================================================================
# The algebra the module docstring proves
# ==========================================================================================


def test_expected_costs_match_the_stated_cost_model() -> None:
    """The two arms must be exactly the arithmetic the docstring claims."""
    model = CostModel(review_cost=3.0, chargeback_fee=15.0)
    probability = np.array([0.0, 0.25, 1.0])
    amounts = np.array([100.0, 100.0, 100.0])

    blocked = expected_cost_if_blocked(probability, model)
    allowed = expected_cost_if_allowed(probability, amounts, model)

    # A certainly-legitimate row costs a full review to block and nothing to allow.
    assert blocked[0] == pytest.approx(3.0)
    assert allowed[0] == pytest.approx(0.0)
    # A certainly-fraudulent row costs nothing to block and the amount plus fee to allow.
    assert blocked[2] == pytest.approx(0.0)
    assert allowed[2] == pytest.approx(115.0)
    assert blocked[1] == pytest.approx(0.75 * 3.0)
    assert allowed[1] == pytest.approx(0.25 * 115.0)


def test_break_even_threshold_is_the_root_of_the_treatment_effect() -> None:
    """``p > r/(A+f+r)`` must be exactly where blocking stops costing more than allowing.

    This is the algebraic claim the whole phase rests on. If the break-even formula and the two
    cost arms ever disagree, the ranking is optimising something other than what is reported.
    """
    model = CostModel(review_cost=3.0, chargeback_fee=15.0)
    amounts = np.array([10.0, 100.0, 10_000.0])
    breakeven = amount_aware_threshold(amounts, model)

    blocked = expected_cost_if_blocked(breakeven, model)
    allowed = expected_cost_if_allowed(breakeven, amounts, model)
    assert blocked == pytest.approx(allowed)

    # And the cut-off must fall as the amount rises -- the point of amount-aware thresholding.
    assert breakeven[0] > breakeven[1] > breakeven[2]


def test_a_large_amount_earns_a_block_at_a_far_lower_probability() -> None:
    """The headline mechanism, stated as a test rather than only as prose."""
    model = CostModel()
    small, large = np.array([20.0]), np.array([5_000.0])
    assert amount_aware_threshold(large, model)[0] < amount_aware_threshold(small, model)[0] / 50


def test_decision_cost_arms_are_consistent_with_expected_cost() -> None:
    """``estimate_cost`` must not be able to report an expected cost from neither arm."""
    policy = CostPolicy(
        strategy="plug_in",
        cost_model=CostModel(),
        threshold=0.0,
        threshold_criterion="test",
    )
    blocked = policy.estimate_cost(amount=250.0, fraud_probability=0.4, decision="block")
    allowed = policy.estimate_cost(amount=250.0, fraud_probability=0.4, decision="allow")

    assert blocked.expected_cost == blocked.cost_if_blocked
    assert allowed.expected_cost == allowed.cost_if_allowed
    assert blocked.cost_if_blocked == allowed.cost_if_blocked
    assert blocked.expected_saving_from_blocking == pytest.approx(
        blocked.cost_if_allowed - blocked.cost_if_blocked
    )


def test_review_is_priced_as_block_not_as_allow() -> None:
    """A reviewed transaction does not complete, so it cannot be priced as an allow."""
    policy = CostPolicy(
        strategy="plug_in", cost_model=CostModel(), threshold=0.0, threshold_criterion="test"
    )
    review = policy.estimate_cost(amount=500.0, fraud_probability=0.3, decision="review")
    block = policy.estimate_cost(amount=500.0, fraud_probability=0.3, decision="block")
    assert review.expected_cost == block.expected_cost


def test_estimate_cost_rejects_an_out_of_range_probability() -> None:
    """A probability outside [0, 1] would silently produce a negative expected cost."""
    policy = CostPolicy(
        strategy="plug_in", cost_model=CostModel(), threshold=0.0, threshold_criterion="test"
    )
    with pytest.raises(ValueError, match="must be in"):
        policy.estimate_cost(amount=10.0, fraud_probability=1.4, decision="block")


def test_decision_cost_carries_its_assumptions() -> None:
    """Section 3 of the standards: a cost figure without its assumptions is incomplete."""
    policy = CostPolicy(
        strategy="plug_in", cost_model=CostModel(), threshold=0.0, threshold_criterion="test"
    )
    estimate = policy.estimate_cost(amount=10.0, fraud_probability=0.5, decision="allow")
    assert len(estimate.assumptions) >= 5
    assert estimate.to_audit_dict()["assumptions"] == estimate.assumptions


# ==========================================================================================
# Ranking policies
# ==========================================================================================


def test_the_three_policies_rank_differently_on_the_same_probability() -> None:
    """If they agreed, the phase would have nothing to measure."""
    model = CostModel()
    probability = np.array([0.10, 0.05])
    amounts = np.array([10.0, 5_000.0])
    predicted_loss = probability * (amounts + model.chargeback_fee)

    def order(strategy: str) -> np.ndarray:
        policy = CostPolicy(
            strategy=strategy, cost_model=model, threshold=0.0, threshold_criterion="test"
        )
        return np.argsort(-policy.ranking_score(probability, amounts, predicted_loss))

    # Probability ranks the riskier row first; cost ranks the more expensive one first.
    assert order("probability")[0] == 0
    assert order("plug_in")[0] == 1


def test_plug_in_ranking_is_the_expected_saving_from_blocking() -> None:
    """The ranking score must be the treatment effect, not a proxy for it."""
    model = CostModel()
    probability = np.array([0.2, 0.6])
    amounts = np.array([100.0, 900.0])
    policy = CostPolicy(
        strategy="plug_in", cost_model=model, threshold=0.0, threshold_criterion="test"
    )
    expected = expected_cost_if_allowed(probability, amounts, model) - expected_cost_if_blocked(
        probability, model
    )
    assert policy.ranking_score(probability, amounts) == pytest.approx(expected)


def test_learned_loss_ranking_refuses_to_run_without_predictions() -> None:
    """Falling back to the plug-in silently would make the two policies indistinguishable."""
    policy = CostPolicy(
        strategy="learned_loss", cost_model=CostModel(), threshold=0.0, threshold_criterion="t"
    )
    with pytest.raises(ValueError, match="needs predicted_loss"):
        policy.ranking_score(np.array([0.5]), np.array([10.0]))


def test_unknown_strategy_is_refused() -> None:
    """A typo in a strategy name must not fall through to some default ranking."""
    policy = CostPolicy(
        strategy="probabilty", cost_model=CostModel(), threshold=0.0, threshold_criterion="t"
    )
    with pytest.raises(ValueError, match="unknown strategy"):
        policy.ranking_score(np.array([0.5]), np.array([10.0]))


# ==========================================================================================
# The loss regression, and the guards on how it is fitted
# ==========================================================================================


def test_loss_target_is_zero_for_legitimate_rows(frame: pd.DataFrame) -> None:
    """The regression target must be the loss if allowed, not the label."""
    model = CostModel()
    labels = frame["is_fraud"].to_numpy(dtype=bool)
    amounts = frame["amount"].to_numpy(dtype="float64")
    target = loss_target(labels, amounts, model)

    assert np.all(target[~labels] == 0.0)
    assert target[labels] == pytest.approx(amounts[labels] + model.chargeback_fee)


def test_loss_model_predictions_are_non_negative(frame: pd.DataFrame) -> None:
    """A negative expected loss would silently invert the ranking."""
    model = CostModel()
    target = loss_target(
        frame["is_fraud"].to_numpy(dtype=bool), frame["amount"].to_numpy("float64"), model
    )
    fitted = fit_loss_model(
        _matrix(frame),
        target,
        feature_names=FEATURES,
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        rounds=40,
    )
    assert np.all(fitted.predict(_matrix(frame)) >= 0.0)


def test_loss_model_records_the_fee_its_target_was_built_with(frame: pd.DataFrame) -> None:
    """A model fitted against one fee is meaningless read against another."""
    model = CostModel(chargeback_fee=99.0)
    target = loss_target(
        frame["is_fraud"].to_numpy(dtype=bool), frame["amount"].to_numpy("float64"), model
    )
    fitted = fit_loss_model(
        _matrix(frame),
        target,
        feature_names=FEATURES,
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        rounds=20,
    )
    assert fitted.chargeback_fee == 99.0


def test_out_of_fold_loss_leaves_the_first_block_unscored(frame: pd.DataFrame) -> None:
    """Block 1 has no predecessor to fit on, and must be NaN rather than quietly zero.

    Zero is the modal value of the target, so a bug that filled block 1 with zeros instead of
    NaN would look entirely plausible in every downstream metric.
    """
    model = CostModel()
    target = loss_target(
        frame["is_fraud"].to_numpy(dtype=bool), frame["amount"].to_numpy("float64"), model
    )
    predictions = out_of_fold_loss(
        frame,
        _matrix(frame),
        target,
        feature_names=FEATURES,
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        blocks=4,
    )
    unscored = np.isnan(predictions)
    assert unscored.any(), "block 1 should be unscored"
    assert not unscored.all(), "every block cannot be unscored"
    # The unscored rows must be the earliest ones -- that is what makes it forward-chaining.
    assert frame.loc[unscored, "event_time"].max() < frame.loc[~unscored, "event_time"].min()


def test_forward_chaining_guard_catches_a_shuffled_fold_assignment(frame: pd.DataFrame) -> None:
    """Plant the violation and assert the guard fires.

    ``out_of_fold_loss`` calls ``assert_forward_chaining`` on every fold rather than trusting
    the block assignment it just built. The violation has to be planted at that call, not by
    shuffling the frame: ``time_blocks`` recomputes the assignment from the event times it is
    handed, so shuffling those produces a different assignment that is still self-consistent.
    The failure the guard actually protects against is an assignment that disagrees with the
    timeline it claims to describe, and the contamination that creates is invisible in every
    metric the run goes on to report.
    """
    from dataclasses import replace

    times = frame["event_time"]
    blocks = time_blocks(times, block_count=4)
    shuffled = replace(blocks, index=np.random.default_rng(0).permutation(blocks.index))

    with pytest.raises(ValueError, match="not forward-chaining"):
        assert_forward_chaining(shuffled, times, 3)


def test_forward_chaining_guard_passes_an_honest_assignment(frame: pd.DataFrame) -> None:
    """The paired half: a guard that rejects the real assignment too would block every run."""
    times = frame["event_time"]
    blocks = time_blocks(times, block_count=4)
    for block in range(2, blocks.block_count + 1):
        assert_forward_chaining(blocks, times, block)


def test_out_of_fold_loss_is_reproducible(frame: pd.DataFrame) -> None:
    """The same frame, twice, must produce identical out-of-fold predictions.

    This is the seed rule from CLAUDE.md made checkable. Note what it deliberately does *not*
    claim: the predictions are reproducible given the same rows **in the same order**, not
    invariant to row order. ``LOSS_PARAMS`` enables bagging and feature sampling, and both draw
    against row position, so a permuted training set fits a slightly different booster. That is
    a property of bagged gradient boosting rather than a defect here -- with sampling disabled
    the two agree exactly -- but it means the on-disk parquet order is part of what makes a run
    reproducible, and it is recorded in ``app.models.causal_cost`` rather than left to be
    rediscovered.
    """
    model = CostModel()
    target = loss_target(
        frame["is_fraud"].to_numpy(dtype=bool), frame["amount"].to_numpy("float64"), model
    )
    kwargs = {
        "feature_names": FEATURES,
        "seed": SEED,
        "chargeback_fee": model.chargeback_fee,
        "blocks": 4,
    }
    first = out_of_fold_loss(frame, _matrix(frame), target, **kwargs)
    second = out_of_fold_loss(frame, _matrix(frame), target, **kwargs)

    scored = ~np.isnan(first)
    assert scored.any()
    assert np.array_equal(first[scored], second[scored])
    assert np.array_equal(np.isnan(first), np.isnan(second))


def test_loss_model_would_notice_a_planted_leak(frame: pd.DataFrame) -> None:
    """Plant the label in the feature matrix and assert the model becomes implausibly good.

    Not a guard on production code -- it is the calibration for the leak-suspicion wire. If a
    planted, perfectly-predictive column did *not* move the fit, the fit is not reading its
    features at all and every other test here is measuring nothing.
    """
    model = CostModel()
    labels = frame["is_fraud"].to_numpy(dtype=bool)
    amounts = frame["amount"].to_numpy("float64")
    target = loss_target(labels, amounts, model)

    clean = fit_loss_model(
        _matrix(frame),
        target,
        feature_names=FEATURES,
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        rounds=60,
    )
    leaked_matrix = np.column_stack([_matrix(frame), target])
    leaked = fit_loss_model(
        leaked_matrix,
        target,
        feature_names=(*FEATURES, "PLANTED_LEAK"),
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        rounds=60,
    )
    clean_error = float(np.mean(np.abs(clean.predict(_matrix(frame)) - target)))
    leaked_error = float(np.mean(np.abs(leaked.predict(leaked_matrix) - target)))
    assert leaked_error < clean_error / 2, (
        "a planted copy of the target did not improve the fit, so the regression is not "
        "reading its feature matrix"
    )


# ==========================================================================================
# Off-policy evaluation
# ==========================================================================================


def test_realised_cost_matches_the_shared_cost_module(frame: pd.DataFrame) -> None:
    """The OPE arithmetic and ``app.ml.cost`` must never drift apart.

    Two implementations of one cost model is how a phase ends up reporting two different
    numbers for the same thing.
    """
    from app.ml.cost import cost_at_threshold

    model = CostModel()
    labels = frame["is_fraud"].to_numpy(dtype=bool)
    amounts = frame["amount"].to_numpy("float64")
    rng = np.random.default_rng(3)
    scores = rng.random(len(frame))

    threshold = 0.7
    assert float(np.sum(realised_cost(labels, amounts, scores >= threshold, model))) == (
        pytest.approx(cost_at_threshold(labels, scores, amounts, threshold, model).total_cost)
    )


def test_true_policy_cost_is_exact_not_estimated() -> None:
    """Hand-computable case, so the ground truth the estimators are scored against is checked."""
    model = CostModel(review_cost=3.0, chargeback_fee=15.0)
    labels = np.array([True, False, True, False])
    amounts = np.array([100.0, 50.0, 200.0, 70.0])
    # Block the two frauds and one legitimate row.
    decisions = np.array([True, True, False, False])
    # blocked fraud 0, blocked legit 3.0, allowed fraud 215.0, allowed legit 0 -> 218 / 4
    assert true_policy_cost(labels, amounts, decisions, model) == pytest.approx(218.0 / 4)


def test_positivity_guard_fires_on_a_deterministic_propensity() -> None:
    """Plant the violation and assert it fires.

    This is the ``isFlaggedFraud`` failure mode exactly: a rule that acts deterministically has
    propensities of 0 and 1, ``1/e(x)`` is undefined, and the resulting estimate is arbitrary
    rather than merely imprecise.
    """
    deterministic = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="positivity violated"):
        assert_positivity(deterministic)


def test_positivity_guard_accepts_a_genuinely_stochastic_policy() -> None:
    """The paired half: a guard that fires on everything is equally useless."""
    assert_positivity(np.array([0.05, 0.5, 0.95]))
    assert_positivity(np.array([POSITIVITY_FLOOR * 10, 1.0 - POSITIVITY_FLOOR * 10]))


def test_logging_policy_refuses_a_deterministic_propensity_at_construction() -> None:
    """The guard must be on the type, not only on the estimators that consume it."""
    with pytest.raises(ValueError, match="positivity violated"):
        LoggingPolicy(
            blocked=np.array([True, False]),
            propensity=np.array([1.0, 0.0]),
            description="a deterministic rule, like isFlaggedFraud",
        )


def test_logging_policy_refuses_mismatched_shapes() -> None:
    """A propensity that does not line up with its decisions silently reweights wrong rows."""
    with pytest.raises(ValueError, match="disagree"):
        LoggingPolicy(
            blocked=np.array([True, False, True]),
            propensity=np.array([0.5, 0.5]),
            description="mismatched",
        )


def test_simulated_policy_is_stochastic_everywhere() -> None:
    """The simulation exists to be invertible; a floor and ceiling are what make it so."""
    rng = np.random.default_rng(5)
    policy = simulate_logging_policy(rng.random(2_000), floor=0.05, ceiling=0.60)
    assert policy.propensity.min() >= 0.05
    assert policy.propensity.max() <= 0.60
    assert "SIMULATED" in policy.description


def test_simulated_policy_refuses_an_impossible_band() -> None:
    """A floor above its ceiling is a configuration error, not something to interpolate."""
    with pytest.raises(ValueError, match="floor < ceiling"):
        simulate_logging_policy(np.array([0.5]), floor=0.8, ceiling=0.2)


def test_ipw_recovers_the_truth_under_a_known_propensity() -> None:
    """The estimator's headline property, checked rather than asserted.

    IPW is unbiased when the propensity is correct. Here it is correct by construction, so this
    is a test of the implementation -- and the only setting in which any of these estimators
    can be checked at all, since this data records no real actions.
    """
    rng = np.random.default_rng(11)
    n = 60_000
    scores = rng.random(n)
    labels = rng.random(n) < (0.02 + 0.15 * scores)
    # Bounded amounts: a lognormal tail makes the IPW variance dominate at any feasible n,
    # which is a fact about heavy tails rather than about the estimator.
    amounts = 50.0 + 100.0 * rng.random(n)
    model = CostModel()

    logged = simulate_logging_policy(scores, seed=1)
    decisions = scores >= 0.9
    observed = realised_cost(labels, amounts, logged.blocked, model)

    truth = true_policy_cost(labels, amounts, decisions, model)
    assert ipw(observed, logged, decisions) == pytest.approx(truth, rel=0.05)


def test_doubly_robust_survives_a_wrong_outcome_model() -> None:
    """The property the name refers to: right propensity is enough, even with a broken model.

    The direct method is handed a deliberately useless outcome model and must be badly wrong;
    the doubly robust estimator is handed the *same* useless model and must still land near the
    truth, because its correction term is weighted by a propensity that is correct.
    """
    rng = np.random.default_rng(13)
    n = 60_000
    scores = rng.random(n)
    labels = rng.random(n) < (0.02 + 0.15 * scores)
    amounts = 50.0 + 100.0 * rng.random(n)
    model = CostModel()

    logged = simulate_logging_policy(scores, seed=2)
    decisions = scores >= 0.9
    observed = realised_cost(labels, amounts, logged.blocked, model)
    truth = true_policy_cost(labels, amounts, decisions, model)

    # A deliberately wrong outcome model: constant, and an order of magnitude out.
    wrong_blocked = np.full(n, 50.0)
    wrong_allowed = np.full(n, 50.0)

    direct = direct_method(wrong_blocked, wrong_allowed, decisions)
    robust = doubly_robust(observed, wrong_blocked, wrong_allowed, logged, decisions)

    assert abs(direct - truth) > abs(robust - truth), (
        "the doubly robust correction did not repair a broken outcome model, which is the one "
        "property that distinguishes it from the direct method"
    )
    assert robust == pytest.approx(truth, rel=0.10)


def test_ope_report_renders_its_caveat() -> None:
    """The simulated-policy caveat must travel with the numbers, not sit in a docstring."""
    rng = np.random.default_rng(17)
    n = 2_000
    scores = rng.random(n)
    labels = rng.random(n) < 0.05
    amounts = rng.lognormal(4.0, 1.0, n)
    model = CostModel()
    logged = simulate_logging_policy(scores, seed=3)
    report = evaluate_policy(
        "test policy",
        labels,
        amounts,
        scores >= 0.9,
        expected_cost_if_blocked(scores, model),
        expected_cost_if_allowed(scores, amounts, model),
        logged,
        model,
    )
    assert "SIMULATED" in report.to_dict()["caveat"]
    assert "not a causal effect" in report.to_dict()["caveat"]
    assert "true cost per unit" in report.render()


# ==========================================================================================
# The shared measurement additions this phase depends on
# ==========================================================================================


def test_cost_curve_value_recall_reaches_one_when_everything_is_flagged() -> None:
    """The value curve must be a genuine share of the fraud value available."""
    rng = np.random.default_rng(19)
    n = 1_000
    labels = rng.random(n) < 0.1
    scores = rng.random(n)
    amounts = rng.lognormal(4.0, 1.0, n)
    curve = cost_curve(labels, scores, amounts, CostModel())

    assert curve.value_recall[0] == pytest.approx(0.0)
    assert curve.value_recall[-1] == pytest.approx(1.0)
    assert np.all(np.diff(curve.value_recall) >= -1e-12), "value recall must be monotone"
    assert curve.total_positive_value == pytest.approx(float(np.sum(amounts[labels])))


def test_value_recall_agrees_with_a_hand_computed_confusion() -> None:
    """The shared helper must not disagree with the confusion matrix at the same threshold."""
    rng = np.random.default_rng(23)
    n = 5_000
    labels = rng.random(n) < 0.08
    scores = rng.random(n) + 0.3 * labels
    amounts = rng.lognormal(4.0, 1.0, n)

    recall = value_recall_at_flag_rate(labels, scores, amounts, 0.01, CostModel())
    expected = confusion_at_threshold(labels, scores, recall.threshold)
    assert recall.confusion.to_dict() == expected.to_dict()
    assert recall.caught_value + recall.missed_value == pytest.approx(recall.total_value)


def test_bootstrap_cost_delta_is_zero_against_itself() -> None:
    """A model compared with itself must produce an interval containing zero.

    The pairing check. If the two arms were scored on different resamples the interval would be
    wide and centred anywhere, and every comparison this phase reports would be meaningless.
    """
    rng = np.random.default_rng(29)
    n = 4_000
    labels = rng.random(n) < 0.08
    scores = rng.random(n) + 0.3 * labels
    amounts = rng.lognormal(4.0, 1.0, n)

    low, high = bootstrap_cost_delta(
        labels, scores, scores, amounts, CostModel(), flag_rate=0.01, resamples=120, seed=0
    )
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_bootstrap_detects_a_genuinely_cheaper_policy() -> None:
    """The paired half: an interval that never excludes zero cannot report a finding.

    A cost-ranked policy is constructed to be genuinely better by value, and the interval on the
    cost difference must exclude zero.
    """
    rng = np.random.default_rng(31)
    n = 40_000
    model = CostModel()
    # A *calibrated* probability, and amounts independent of the label. Cost-aware ranking only
    # pays when the probability is informative: multiplying a noisy score by an independent
    # amount adds more noise than signal, which is itself worth knowing and is why this
    # construction is deliberate rather than arbitrary.
    probability = 0.02 + 0.5 * rng.random(n)
    labels = rng.random(n) < probability
    amounts = rng.lognormal(4.0, 1.2, n)
    cost_ranked = probability * (amounts + model.chargeback_fee)

    low, high = bootstrap_cost_delta(
        labels, cost_ranked, probability, amounts, model, flag_rate=0.01, resamples=200, seed=0
    )
    assert high < 0.0, "cost-aware ranking should be measurably cheaper on this construction"

    value_low, value_high = bootstrap_value_recall_delta(
        labels, cost_ranked, probability, amounts, flag_rate=0.01, resamples=200, seed=0
    )
    assert value_low > 0.0, "and it should capture measurably more fraud value"


def test_format_threshold_does_not_collapse_distinct_values() -> None:
    """The Phase 5 trap: ``%.6f`` printed a 0.00175-wide band as two identical strings."""
    assert format_threshold(0.6035219404714103) != format_threshold(0.6052732101577928)
    assert format_threshold(1e-9) != format_threshold(2e-9)
    assert format_threshold(float("inf")) == "never flag"
    # Values that fixed notation represents faithfully keep it, because it reads better.
    assert format_threshold(0.5) == "0.500000"


def test_cnp_regime_prices_a_false_positive_far_above_the_default() -> None:
    """The named alternative regime must be a real alternative, not a relabelled default."""
    default, cnp = CostModel(), cnp_cost_model()
    assert cnp.review_cost > 10 * default.review_cost
    assert cnp.chargeback_fee > 10 * default.chargeback_fee


# ==========================================================================================
# Persistence
# ==========================================================================================


def test_policy_round_trips_through_save_and_load(tmp_path, frame: pd.DataFrame) -> None:
    """A reloaded policy must decide identically, or the registry entry describes nothing."""
    model = CostModel()
    target = loss_target(
        frame["is_fraud"].to_numpy(dtype=bool), frame["amount"].to_numpy("float64"), model
    )
    loss_model = fit_loss_model(
        _matrix(frame),
        target,
        feature_names=FEATURES,
        seed=SEED,
        chargeback_fee=model.chargeback_fee,
        rounds=30,
    )
    policy = CostPolicy(
        strategy="learned_loss",
        cost_model=model,
        threshold=12.5,
        threshold_criterion="a 1% cap on V-late",
        loss_model=loss_model,
    )
    policy.save("causal-cost-test-20260101t000000z", tmp_path)
    reloaded = CostPolicy.load("causal-cost-test-20260101t000000z", tmp_path)

    assert reloaded.strategy == policy.strategy
    assert reloaded.threshold == policy.threshold
    assert reloaded.cost_model == policy.cost_model
    assert reloaded.loss_model is not None
    matrix = _matrix(frame)
    assert reloaded.loss_model.predict(matrix) == pytest.approx(loss_model.predict(matrix))


def test_save_refuses_a_path_traversing_model_id(tmp_path) -> None:
    """Model ids are file content, not trusted identifiers.

    ``artifact_path`` closes a real Windows pathlib hole -- ``Path("a/b") / "C:evil.json"``
    discards the base entirely -- and it reached two tiers before it was fixed. This phase must
    go through the same guard rather than joining paths itself.
    """
    policy = CostPolicy(
        strategy="plug_in", cost_model=CostModel(), threshold=1.0, threshold_criterion="t"
    )
    with pytest.raises(ValueError):
        policy.save("../../escaped", tmp_path)


def test_saved_sidecar_carries_the_cost_assumptions(tmp_path) -> None:
    """An artefact whose cost basis is not recorded cannot be audited later."""
    import json

    policy = CostPolicy(
        strategy="plug_in",
        cost_model=CostModel(),
        threshold=1.0,
        threshold_criterion="a 1% cap on V-late",
    )
    sidecar = policy.save("causal-cost-plain-20260101t000000z", tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["cost_model"]["review_cost"] == CostModel().review_cost
    assert len(payload["assumptions"]) >= 5
    assert payload["loss_model"] is None


def test_decision_cost_dict_is_json_serialisable() -> None:
    """It goes into the audit trail, so it must survive serialisation."""
    import json

    estimate = DecisionCost(
        expected_cost=1.5,
        cost_if_blocked=1.5,
        cost_if_allowed=20.0,
        fraud_probability=0.25,
        amount=65.0,
        decision="block",
        assumptions=["an assumption"],
    )
    assert json.loads(json.dumps(estimate.to_audit_dict()))["decision"] == "block"


# ==========================================================================================
# Regressions from the Phase 6 gates
# ==========================================================================================


def test_result_confusion_and_cost_come_from_one_threshold() -> None:
    """A result block must never pair a confusion matrix with a cost from a different cut.

    The bug of record for this phase. The shipped result rendered its matrix at the V-late
    threshold (1,203 rows flagged) and its cost at the test 1% quantile (886 flagged). Both
    numbers were individually correct and the block was a self-contradiction, which is worse
    than either being wrong, and it was caught by eye rather than by a test.

    The invariant is arithmetic and cheap: whatever threshold a result reports, the rows its
    matrix calls flagged (FP + TP) must equal the rows its cost estimate calls flagged.
    """
    rng = np.random.default_rng(41)
    n = 4_000
    labels = rng.random(n) < 0.06
    scores = rng.random(n) + 0.3 * labels
    amounts = rng.lognormal(4.0, 1.0, n)
    model = CostModel()

    # Two genuinely different cuts, so a mismatch would show.
    shipped_threshold = 0.82
    assert threshold_for_flag_rate(scores, 0.01) != shipped_threshold

    shipped = value_recall_at_threshold(labels, scores, amounts, shipped_threshold, model)
    result = evaluate(
        "policy",
        "test",
        labels,
        scores,
        threshold=shipped_threshold,
        threshold_criterion="a V-late cut transferred to test",
        cost=shipped.estimate,
    )

    assert result.false_positive_cost is not None
    flagged_by_matrix = result.confusion.false_positives + result.confusion.true_positives
    assert flagged_by_matrix == result.false_positive_cost.flagged
    assert result.threshold == result.false_positive_cost.threshold


def test_sensitivity_recomputes_the_score_under_each_scaled_cost_model() -> None:
    """Scaling the cost model must change a cost-aware policy's ranking, not just its threshold.

    ``plug_in`` ranks by ``p*(A+f) - (1-p)*r``, so the score itself embeds both parameters.
    Sweeping the cost model while reusing scores computed at 1.0x re-thresholds a policy nobody
    ran. Harmless for a probability ranking, wrong for this one -- so the guard has to show the
    two differ.
    """
    rng = np.random.default_rng(43)
    n = 500
    probability = rng.random(n) * 0.4
    amounts = rng.lognormal(4.0, 1.0, n)
    base = CostModel()
    scaled = base.scaled_by(50.0)

    at_base = scores_under("plug_in", base, probability, amounts, None, None)
    at_scaled = scores_under("plug_in", scaled, probability, amounts, None, None)

    assert not np.allclose(at_base, at_scaled), "the scaled policy must not reuse 1.0x scores"
    # A probability ranking carries no cost inside it, so it is genuinely invariant -- which is
    # why this was safe in Phases 2-5 and is not safe here.
    assert np.allclose(
        scores_under("probability", base, probability, amounts, None, None),
        scores_under("probability", scaled, probability, amounts, None, None),
    )


def test_learned_loss_prediction_is_corrected_for_a_different_fee() -> None:
    """A loss model fitted at one fee must be re-priced, not read raw, under another.

    Its target was ``Y*(A+f)``, so ``E[Y*(A+f2)] = E[Y*(A+f)] + (f2-f)*p``. Without the shift
    the card-not-present row prices at 500 a model fitted at 15, and the row is not the policy
    it claims to be.
    """
    rng = np.random.default_rng(47)
    n = 300
    probability = rng.random(n) * 0.3
    amounts = rng.lognormal(4.0, 1.0, n)
    predicted = probability * (amounts + 15.0)
    fitted = LossModel(
        booster=None, feature_names=("f",), best_iteration=1, chargeback_fee=15.0
    )
    cnp = cnp_cost_model()

    at_cnp = scores_under("learned_loss", cnp, probability, amounts, predicted, fitted)
    expected_loss = predicted + (cnp.chargeback_fee - 15.0) * probability
    expected = expected_loss - (1.0 - probability) * cnp.review_cost
    assert at_cnp == pytest.approx(expected)


def test_estimate_cost_refuses_a_learned_loss_policy_without_a_prediction() -> None:
    """The audit path must refuse what the ranking path refuses.

    Falling back to the plug-in here would record a cost against the decision that is not the
    cost the decision was made on -- the exact drift the class exists to prevent.
    """
    policy = CostPolicy(
        strategy="learned_loss",
        cost_model=CostModel(),
        threshold=1.0,
        threshold_criterion="t",
        loss_model=None,
    )
    with pytest.raises(ValueError, match="needs predicted_loss"):
        policy.estimate_cost(amount=100.0, fraud_probability=0.4, decision="block")


def test_estimate_cost_rejects_an_out_of_range_amount() -> None:
    """A negative amount would invert the ranking score and switch blocking off.

    ``cost_if_allowed = p*(A+f)`` goes negative, the expected saving falls far below any
    threshold, and the transaction is allowed however fraudulent the model believes it to be.
    One scoring call with a negative amount disables the cost layer for that row.
    """
    policy = CostPolicy(
        strategy="plug_in", cost_model=CostModel(), threshold=1.0, threshold_criterion="t"
    )
    with pytest.raises(ValueError, match="amount must be in"):
        policy.estimate_cost(amount=-1_000_000.0, fraud_probability=0.99, decision="allow")
    with pytest.raises(ValueError, match="amount must be in"):
        policy.estimate_cost(amount=1e18, fraud_probability=0.5, decision="allow")
    # The boundaries themselves are legal.
    policy.estimate_cost(amount=0.0, fraud_probability=0.5, decision="allow")


def test_negative_amount_would_have_suppressed_blocking() -> None:
    """Show the vulnerability the guard closes, rather than only asserting the guard exists."""
    model = CostModel()
    policy = CostPolicy(
        strategy="plug_in", cost_model=model, threshold=90.85, threshold_criterion="t"
    )
    honest = policy.ranking_score(np.array([0.99]), np.array([5_000.0]))
    poisoned = policy.ranking_score(np.array([0.99]), np.array([-1_000_000.0]))

    assert honest[0] >= policy.threshold, "a large, near-certain fraud should be blocked"
    assert poisoned[0] < policy.threshold, "the negative amount is what the guard prevents"


def test_decision_cost_audit_dict_is_not_named_like_an_api_shape() -> None:
    """The per-decision cost is an evasion oracle and must not read as a response body.

    The sign of ``expected_saving_from_blocking`` is the decision boundary, and with the
    probability and the two arms a caller recovers the whole cost matrix. ``app.core.audit``
    already carries this warning on ``top_features``; this object is strictly worse.
    """
    estimate = DecisionCost(
        expected_cost=1.5,
        cost_if_blocked=1.5,
        cost_if_allowed=20.0,
        fraud_probability=0.25,
        amount=65.0,
        decision="block",
        assumptions=["an assumption"],
    )
    assert not hasattr(estimate, "to_dict"), "to_dict reads as an API shape; use to_audit_dict"
    payload = estimate.to_audit_dict()
    assert payload["decision"] == "block"
    assert "never a response body" in (DecisionCost.to_audit_dict.__doc__ or "").lower()
