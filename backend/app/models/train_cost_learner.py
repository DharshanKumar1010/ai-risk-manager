"""Train, evaluate and register the RiskIQ causal cost layer (Phase 6).

**The question this phase asks.** Phase 5 found the meta-learner catching less fraud by count
than Tier-1 and more by value, and therefore costing less at a matched flag rate -- but reported
that advantage as a bare point estimate, because the project had an interval for a PR-AUC
difference and none for a cost one. BUILD_LOG recorded the gap and handed it here.

So this run compares three ranking policies, all reading the *same* calibrated Tier-1
probability, differing only in what they rank by:

* ``probability``   -- rank by ``p(x)``. The Phase 2 baseline.
* ``plug_in``       -- rank by expected cost saving, ``p(x)*(A+f) - (1-p(x))*r``.
* ``learned_loss``  -- rank by the loss regression's estimate in place of the plug-in term.

Holding the score fixed is the point: any difference between them is attributable to cost
weighting rather than to one policy having a better underlying detector.

**Pre-registered, before test was read.** The headline is the cost difference at a matched 1%
flag rate, with a paired bootstrap interval. *If the interval straddles zero, this phase reports
a tie*, however far apart the point estimates are. The brief anticipated a 5-15% reduction; that
was a hypothesis, and Phase 5's own cost advantage of 1.7% turned on 1.9 points of value recall
on a quantity dominated by a handful of large transactions.

**Measurement discipline.** The Tier-1 calibrator is fitted on V-fit. Every threshold is chosen
on V-late, the slice closest to test. Test is scored exactly once, after all three operating
points are fixed, and selects nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.config import Settings, get_settings
from app.data.raw_spec import SourceDataset
from app.data.splitting import find_boundary_overlaps, validation_slices
from app.ml.cost import (
    REVIEW_COST_MULTIPLES,
    SENSITIVITY_FACTORS,
    CostModel,
    SensitivityRow,
    ValueRecall,
    choose_threshold_by_cost,
    cnp_cost_model,
    cost_curve,
    render_sensitivity,
    threshold_for_flag_rate,
    value_recall_at_flag_rate,
)
from app.ml.evaluation import (
    EvaluationResult,
    bootstrap_cost_delta,
    bootstrap_pr_auc,
    bootstrap_value_recall_delta,
    evaluate,
    format_threshold,
)
from app.ml.ope import OpeReport, evaluate_policy, simulate_logging_policy
from app.ml.registry import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REGISTRY_PATH,
    RegistryEntry,
    append_entry,
    build_model_id,
)
from app.models.causal_cost import (
    LOSS_OOF_BLOCKS,
    CostPolicy,
    LossModel,
    amount_aware_threshold,
    expected_cost_if_allowed,
    expected_cost_if_blocked,
    fit_loss_model,
    loss_target,
    out_of_fold_loss,
)
from app.models.meta_features import load_registered_tier1
from app.models.meta_learner import SigmoidCalibrator

logger = logging.getLogger("riskiq.cost")

#: Seed for every fit and every bootstrap in this phase. Logged on entry, per CLAUDE.md.
RANDOM_SEED = 42

#: The review-capacity ceiling every phase in this project reports against.
MAX_REVIEW_FLAG_RATE = 0.01

#: The three policies, in report order.
STRATEGIES: tuple[str, ...] = ("probability", "plug_in", "learned_loss")

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


def load_splits(
    processed_dir: Path, source: SourceDataset, sample: int | None
) -> dict[str, pd.DataFrame]:
    """Load the three chronological splits, refusing anything but IEEE-CIS.

    Raises:
        FileNotFoundError: If a split parquet is missing.
        RuntimeError: If the splits on disk are not strictly time-ordered.
        ValueError: For any corpus other than ``ieee_cis``.
    """
    if source != "ieee_cis":
        raise ValueError(
            f"Phase 6 runs on ieee_cis only, got {source!r}. PaySim amounts are simulator units "
            "with no currency interpretation, so a cost figure on that corpus is uninterpretable."
        )
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        path = processed_dir / f"{source}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m app.data.pipeline` to build the feature store."
            )
        frame = pd.read_parquet(path)
        frames[split] = frame.head(sample).copy() if sample is not None else frame
        logger.info("%s/%s: loaded %d rows", source, split, len(frames[split]))

    combined = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()], ignore_index=True
    )
    overlaps = find_boundary_overlaps(combined)
    if overlaps:
        raise RuntimeError(f"{source}: split boundaries overlap on disk -- {overlaps}")
    return frames


def _labels(frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Return the binary fraud labels."""
    return frame["is_fraud"].to_numpy(dtype=bool)


def _amounts(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return transaction amounts, which set what a miss costs."""
    return frame["amount"].to_numpy(dtype="float64")


@dataclass(frozen=True)
class PolicyResult:
    """One ranking policy, measured at its shipped point and at the matched flag rate."""

    strategy: str
    policy: CostPolicy
    result: EvaluationResult
    shipped: ValueRecall
    matched: ValueRecall
    optimum: ValueRecall

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape."""
        return {
            "strategy": self.strategy,
            "heldout_test": self.result.to_dict(),
            "shipped_point": self.shipped.to_dict(),
            "matched_flag_rate": self.matched.to_dict(),
            "cost_optimal_point": self.optimum.to_dict(),
        }


@dataclass
class CostRunReport:
    """Everything one Phase 6 run produced."""

    model_id: str
    cost_model: CostModel
    policies: list[PolicyResult]
    deltas: list[dict[str, Any]]
    ope: list[OpeReport]
    calibration: dict[str, Any]
    sensitivity: list[dict[str, Any]]
    sensitivity_rows: list[Any]
    cnp_regime: list[dict[str, Any]]
    cnp_deltas: list[dict[str, Any]]
    median_amount: float
    slices: list[dict[str, Any]]
    audit_rows: dict[str, list[dict[str, Any]]]
    curve_path: Path | None
    tier1_model_id: str
    feature_version: str
    training_window: dict[str, str]
    loss_model: LossModel
    notes: list[str] = field(default_factory=list)

    def by_strategy(self, strategy: str) -> PolicyResult:
        """Return one policy's result by name."""
        for policy in self.policies:
            if policy.strategy == strategy:
                return policy
        raise KeyError(strategy)


def tier1_logit(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the log-odds of a Tier-1 score, which is the scale Platt scaling operates on."""
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def calibrate_tier1(
    fit_scores: npt.NDArray[np.float64],
    fit_labels: npt.NDArray[np.bool_],
) -> SigmoidCalibrator:
    """Platt-scale Tier-1's score into a calibrated probability, fitted on V-fit.

    Tier-1 trains with ``objective: binary`` under an identity normaliser, so its score is
    already on a probability scale -- but "on a probability scale" is not the same as calibrated,
    and this corpus's base rate drifts visibly across the six months it spans. Every figure this
    phase produces multiplies a probability by an amount, so a score that is merely monotone in
    risk is not good enough: a 10% overstatement of ``p`` overstates every expected loss by 10%
    and moves every break-even threshold with it.

    Fitted with a plain logistic regression on the log-odds rather than through
    :func:`app.models.meta_learner.fit_calibrator`, which wraps its input in an
    ``xgboost.DMatrix`` and so cannot accept a score vector that no XGBoost booster produced.
    The arithmetic is identical -- Platt scaling *is* a logistic regression on the margin -- and
    the result is the same two-float :class:`SigmoidCalibrator` the rest of the project uses.

    Fitted on V-fit and never on V-late, because V-late is where the operating points are chosen.
    """
    from sklearn.linear_model import LogisticRegression

    margin = tier1_logit(fit_scores).reshape(-1, 1)
    fitted = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    fitted.fit(margin, fit_labels.astype("int8"))
    # sklearn parameterises P = sigmoid(w*m + c); SigmoidCalibrator uses P = 1/(1+exp(a*m + b)).
    # The two agree when a = -w and b = -c.
    calibrator = SigmoidCalibrator(
        a=-float(fitted.coef_[0][0]), b=-float(fitted.intercept_[0])
    )
    logger.info("Tier-1 Platt calibration: a=%.6f b=%.6f", calibrator.a, calibrator.b)
    return calibrator


def scores_under(
    strategy: str,
    variant: CostModel,
    probability: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    predicted_loss: npt.NDArray[np.float64] | None,
    loss_model: LossModel | None,
) -> npt.NDArray[np.float64]:
    """Return the ranking score a policy produces under a cost model other than the fitted one.

    Two corrections that are easy to miss, and both were missed on the first pass.

    **The plug-in score embeds the cost parameters.** ``p*(A+f) - (1-p)*r`` moves when ``r`` or
    ``f`` moves, so a sensitivity sweep that scales the cost model while reusing scores computed
    at 1.0x is not evaluating the scaled policy at all -- it re-thresholds the original one
    against a different cost function. Harmless in Phases 2-5, where the score is a probability
    and carries no cost inside it. Wrong here.

    **The loss regression was fitted against one particular fee.** Its target was
    ``Y*(A+f_fitted)``, so under a different fee the prediction needs correcting:
    ``E[Y*(A+f2)] = E[Y*(A+f)] + (f2 - f) * p``. Without it the card-not-present row for
    ``learned_loss`` prices at 500 a model fitted at 15, and the row is not the policy it claims
    to be. :attr:`LossModel.chargeback_fee` is recorded for exactly this check.
    """
    adjusted = predicted_loss
    if strategy == "learned_loss":
        if predicted_loss is None or loss_model is None:
            raise ValueError("learned_loss needs both a prediction vector and its loss model")
        shift = (variant.chargeback_fee - loss_model.chargeback_fee) * probability
        adjusted = predicted_loss + shift
    return CostPolicy(
        strategy=strategy,
        cost_model=variant,
        threshold=float("inf"),
        threshold_criterion="",
        loss_model=loss_model if strategy == "learned_loss" else None,
    ).ranking_score(probability, amounts, adjusted)


def build_policies(
    late: pd.DataFrame,
    late_probability: npt.NDArray[np.float64],
    late_loss: npt.NDArray[np.float64],
    cost_model: CostModel,
    loss_model: LossModel,
) -> dict[str, CostPolicy]:
    """Choose each policy's operating point on V-late, under the shared capacity cap.

    All three are held to the same 1% review ceiling. Comparing a cost-optimal threshold against
    a capacity-capped one would be comparing two products, not two rankings -- and the
    unconstrained cost optimum on this corpus flags far more traffic than any queue could absorb,
    which ``CostModel.assumptions`` names as the assumption that bites hardest.
    """
    late_amounts = _amounts(late)
    policies: dict[str, CostPolicy] = {}
    for strategy in STRATEGIES:
        draft = CostPolicy(
            strategy=strategy,
            cost_model=cost_model,
            threshold=float("inf"),
            threshold_criterion="",
            loss_model=loss_model if strategy == "learned_loss" else None,
        )
        scores = draft.ranking_score(late_probability, late_amounts, late_loss)
        threshold = threshold_for_flag_rate(scores, MAX_REVIEW_FLAG_RATE)
        policies[strategy] = CostPolicy(
            strategy=strategy,
            cost_model=cost_model,
            threshold=threshold,
            threshold_criterion=(
                f"flagging at most {MAX_REVIEW_FLAG_RATE:.1%} of V-late validation traffic, "
                "transferred to test unchanged"
            ),
            loss_model=loss_model if strategy == "learned_loss" else None,
        )
        logger.info(
            "%s: V-late threshold %s (flag rate %.4f%%)",
            strategy,
            format_threshold(threshold),
            100 * float(np.mean(scores >= threshold)),
        )
    return policies


# ==========================================================================================
# The run
# ==========================================================================================


def choose_cost_optimal(
    late: pd.DataFrame,
    scores: npt.NDArray[np.float64],
    cost_model: CostModel,
) -> float:
    """Return the cost-minimising threshold on the V-late ranking scores.

    Chosen on validation, never on test. Reported beside the capacity-capped point rather than
    instead of it, because the unconstrained optimum on this corpus flags far more traffic than
    a review queue could absorb -- the assumption ``CostModel.assumptions`` calls the one that
    bites hardest.
    """
    threshold, _ = choose_threshold_by_cost(_labels(late), scores, _amounts(late), cost_model)
    return threshold


def audit_extremes(
    frame: pd.DataFrame,
    labels: npt.NDArray[np.bool_],
    flagged: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
    count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Return the costliest false positives and false negatives, for the audit trail.

    Required by ml-evaluation-standards section 4: the "what this does NOT catch" section must
    be written from false negatives actually observed, not from imagination. These are the rows
    that section is written from.

    False positives are ranked by amount because that is the customer whose legitimate purchase
    was declined; false negatives by amount because that is the money actually lost.
    """
    false_positive = ~labels & flagged
    false_negative = labels & ~flagged

    def top(mask: npt.NDArray[np.bool_]) -> list[dict[str, Any]]:
        index = np.flatnonzero(mask)
        if index.size == 0:
            return []
        order = index[np.argsort(-amounts[index])][:count]
        return [
            {
                "amount": round(float(amounts[position]), 2),
                "fraud_probability": round(float(probability[position]), 6),
                "product": str(frame["transaction_type"].to_numpy()[position]),
                "device_is_new": bool(frame["device_is_new"].to_numpy()[position]),
                "addr_is_new": bool(frame["addr_is_new"].to_numpy()[position]),
                "account_prior_txn_count": int(
                    frame["account_prior_txn_count"].to_numpy()[position]
                ),
            }
            for position in order
        ]

    return {"false_positives": top(false_positive), "false_negatives": top(false_negative)}


def plot_cost_curve(
    curves: dict[str, Any],
    path: Path,
    cost_model: CostModel,
) -> None:
    """Write the cost curve across the full operating range. Agg backend only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (cost_axis, value_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
    for label, curve in curves.items():
        rate = curve.flagged / curve.flagged[-1] if curve.flagged[-1] else curve.flagged
        cost_axis.plot(100 * rate, 1_000 * curve.costs / curve.flagged[-1], label=label, lw=1.6)
        value_axis.plot(100 * rate, 100 * curve.value_recall, label=label, lw=1.6)
    for axis in (cost_axis, value_axis):
        axis.axvline(100 * MAX_REVIEW_FLAG_RATE, color="0.4", ls="--", lw=1.0)
        axis.set_xscale("log")
        axis.set_xlabel("flag rate (%, log scale)")
        axis.legend(frameon=False, fontsize=9)
        axis.grid(alpha=0.25)
    cost_axis.set_ylabel(f"cost per 1,000 [{cost_model.units}]")
    cost_axis.set_title("Cost across the operating range")
    value_axis.set_ylabel("fraud value captured (%)")
    value_axis.set_title("Value captured across the operating range")
    figure.suptitle(
        "Dashed line: the 1% review-capacity cap. Costs are estimates on stated assumptions.",
        fontsize=9,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run(
    settings: Settings,
    *,
    sample: int | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_path: Path | None = None,
) -> CostRunReport:
    """Fit the cost layer, compare three ranking policies, and score test exactly once."""
    logger.info("random seed = %d", RANDOM_SEED)
    frames = load_splits(settings.processed_data_dir, "ieee_cis", sample)
    train, validation, test = frames["train"], frames["val"], frames["test"]

    registered = load_registered_tier1(artifact_dir, registry_path)
    if registered is None:
        raise RuntimeError(
            "No registered Tier-1 model found. Phase 6 consumes Tier-1's score; run "
            "`python -m app.models.train_tier1` first."
        )
    logger.info("consuming Tier-1 model %s", registered.model_id)

    slices = validation_slices(validation)
    cost_model = CostModel()

    # --- Tier-1 scores, then a calibrator fitted on V-fit alone.
    fit_scores = registered.model.score_frame(slices.fit)
    calibrator = calibrate_tier1(fit_scores, _labels(slices.fit))

    def probability_of(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return calibrator.apply(tier1_logit(registered.model.score_frame(frame)))

    late_probability = probability_of(slices.late)
    test_probability = probability_of(test)

    # --- The loss regression, fitted on train and early-stopped on V-fit.
    feature_names = registered.model.feature_names
    train_matrix = registered.model.prepare(train)
    train_target = loss_target(_labels(train), _amounts(train), cost_model)
    loss_model = fit_loss_model(
        train_matrix,
        train_target,
        feature_names=feature_names,
        seed=RANDOM_SEED,
        chargeback_fee=cost_model.chargeback_fee,
        validation=(
            registered.model.prepare(slices.fit),
            loss_target(_labels(slices.fit), _amounts(slices.fit), cost_model),
        ),
    )
    logger.info(
        "loss regression: %d features, best_iteration=%d",
        len(feature_names),
        loss_model.best_iteration,
    )
    oof_loss = out_of_fold_loss(
        train,
        train_matrix,
        train_target,
        feature_names=feature_names,
        seed=RANDOM_SEED,
        chargeback_fee=cost_model.chargeback_fee,
    )
    scored = ~np.isnan(oof_loss)
    late_loss = loss_model.predict(registered.model.prepare(slices.late))
    test_loss = loss_model.predict(registered.model.prepare(test))

    # --- Operating points, all chosen on V-late.
    policies = build_policies(slices.late, late_probability, late_loss, cost_model, loss_model)

    # --- Test is read from here on, and selects nothing.
    test_labels = _labels(test)
    test_amounts = _amounts(test)
    results: list[PolicyResult] = []
    test_scores: dict[str, npt.NDArray[np.float64]] = {}
    curves: dict[str, Any] = {}
    for strategy in STRATEGIES:
        policy = policies[strategy]
        scores = policy.ranking_score(test_probability, test_amounts, test_loss)
        test_scores[strategy] = scores
        curves[strategy] = cost_curve(test_labels, scores, test_amounts, cost_model)

        late_scores = policy.ranking_score(late_probability, _amounts(slices.late), late_loss)
        optimum_threshold = choose_cost_optimal(slices.late, late_scores, cost_model)

        from app.ml.cost import value_recall_at_threshold

        matched = value_recall_at_flag_rate(
            test_labels, scores, test_amounts, MAX_REVIEW_FLAG_RATE, cost_model
        )
        optimum = value_recall_at_threshold(
            test_labels, scores, test_amounts, optimum_threshold, cost_model
        )
        # The shipped point: the V-late threshold transferred to test unchanged. Its cost has to
        # be measured at that same threshold. Handing `evaluate` the matched-flag-rate cost
        # instead put a confusion matrix from one operating point beside a cost block from
        # another inside a single result -- 1,203 rows flagged in the matrix against 886 in the
        # cost, which reads as a self-contradiction and would have shipped as one.
        shipped = value_recall_at_threshold(
            test_labels, scores, test_amounts, policy.threshold, cost_model
        )
        results.append(
            PolicyResult(
                strategy=strategy,
                policy=policy,
                result=evaluate(
                    f"cost-policy:{strategy}",
                    "test",
                    test_labels,
                    scores,
                    threshold=policy.threshold,
                    threshold_criterion=policy.threshold_criterion,
                    interval=bootstrap_pr_auc(test_labels, scores, seed=RANDOM_SEED),
                    cost=shipped.estimate,
                    notes=[
                        "PR-AUC here ranks by expected cost, not by probability. It is reported "
                        "for continuity with Phases 2-5 and is NOT this phase's headline -- a "
                        "cost-ranked policy is not trying to win a count-based metric, and "
                        "losing one is the expected consequence of ranking by value instead."
                    ]
                    if strategy != "probability"
                    else [],
                ),
                shipped=shipped,
                matched=matched,
                optimum=optimum,
            )
        )

    # --- The headline: paired intervals on the cost and value differences.
    # Each comparison is a PAIRED interval against a named baseline. The third pair is not
    # decoration: reading "the loss regression adds nothing" off two overlapping intervals that
    # share a baseline is not a valid comparison, and that question -- does cost-sensitive
    # *training* beat cost-sensitive *thresholding*? -- is the one BUILD_LOG handed this phase.
    comparisons: tuple[tuple[str, str], ...] = (
        ("plug_in", "probability"),
        ("learned_loss", "probability"),
        ("learned_loss", "plug_in"),
    )
    deltas: list[dict[str, Any]] = []
    for strategy, against in comparisons:
        baseline = test_scores[against]
        baseline_matched = next(r for r in results if r.strategy == against).matched
        cost_interval = bootstrap_cost_delta(
            test_labels,
            test_scores[strategy],
            baseline,
            test_amounts,
            cost_model,
            flag_rate=MAX_REVIEW_FLAG_RATE,
            seed=RANDOM_SEED,
        )
        value_interval = bootstrap_value_recall_delta(
            test_labels,
            test_scores[strategy],
            baseline,
            test_amounts,
            flag_rate=MAX_REVIEW_FLAG_RATE,
            seed=RANDOM_SEED,
        )
        candidate = next(r for r in results if r.strategy == strategy).matched
        cost_point = candidate.estimate.cost_per_1000_units - (
            baseline_matched.estimate.cost_per_1000_units
        )
        tie = cost_interval[0] <= 0.0 <= cost_interval[1]
        deltas.append(
            {
                "policy": strategy,
                "baseline": against,
                "flag_rate": MAX_REVIEW_FLAG_RATE,
                "cost_delta_per_1000": round(cost_point, 4),
                "cost_delta_ci95": [round(value, 4) for value in cost_interval],
                "cost_delta_pct": round(
                    100 * cost_point / baseline_matched.estimate.cost_per_1000_units, 4
                )
                if baseline_matched.estimate.cost_per_1000_units
                else 0.0,
                "value_recall_delta": round(
                    candidate.recall_by_value - baseline_matched.recall_by_value, 6
                ),
                "value_recall_delta_ci95": [round(value, 6) for value in value_interval],
                "verdict": (
                    "TIE -- the interval on the cost difference includes zero"
                    if tie
                    else ("CHEAPER than the baseline" if cost_point < 0 else "MORE EXPENSIVE")
                ),
            }
        )
        logger.info(
            "%s vs %s: cost delta %+.2f per 1,000, CI [%+.2f, %+.2f] -- %s",
            strategy,
            against,
            cost_point,
            cost_interval[0],
            cost_interval[1],
            "tie" if tie else "excludes zero",
        )

    def _matched_threshold(rows: list[PolicyResult], strategy: str) -> float:
        return next(row for row in rows if row.strategy == strategy).matched.threshold

    # --- Off-policy evaluation, against a simulated logging policy with a known propensity.
    logged = simulate_logging_policy(test_probability, seed=RANDOM_SEED)
    blocked_cost = expected_cost_if_blocked(test_probability, cost_model)
    allowed_cost = expected_cost_if_allowed(test_probability, test_amounts, cost_model)
    ope = [
        evaluate_policy(
            f"cost-policy:{strategy} at {MAX_REVIEW_FLAG_RATE:.0%} flag rate",
            test_labels,
            test_amounts,
            test_scores[strategy] >= _matched_threshold(results, strategy),
            blocked_cost,
            allowed_cost,
            logged,
            cost_model,
        )
        for strategy in STRATEGIES
    ]

    # --- Sensitivity, on V-late. Chosen on validation, as required.
    #
    # plug_in is the headline because the collapse proof identifies it as the closed form of the
    # decision rule, not because it won a comparison on test. Where it ties learned_loss the
    # tiebreak is parsimony: the plug-in ships no booster, so its artefact is a JSON file and its
    # load path unpickles nothing.
    headline = "plug_in"
    late_labels, late_amounts = _labels(slices.late), _amounts(slices.late)
    sweeps: tuple[tuple[str, tuple[float, ...], list[CostModel]], ...] = (
        (
            "both costs",
            SENSITIVITY_FACTORS,
            [cost_model.scaled_by(factor) for factor in SENSITIVITY_FACTORS],
        ),
        (
            "review cost only",
            REVIEW_COST_MULTIPLES,
            [
                CostModel(
                    review_cost=cost_model.review_cost * multiple,
                    chargeback_fee=cost_model.chargeback_fee,
                    units=cost_model.units,
                )
                for multiple in REVIEW_COST_MULTIPLES
            ],
        ),
    )
    sensitivity_rows: list[SensitivityRow] = []
    for label, factors, variants in sweeps:
        for factor, variant in zip(factors, variants, strict=True):
            # Recomputed under the scaled model rather than reused from 1.0x -- see
            # scores_under. Reusing it would sweep the threshold of a policy nobody ran.
            variant_scores = scores_under(
                headline, variant, late_probability, late_amounts, late_loss, loss_model
            )
            threshold, estimate = choose_threshold_by_cost(
                late_labels, variant_scores, late_amounts, variant
            )
            sensitivity_rows.append(
                SensitivityRow(factor=factor, threshold=threshold, estimate=estimate, label=label)
            )
    sensitivity = [row.to_dict() for row in sensitivity_rows]

    # --- The CNP regime, re-derived end to end rather than rescaled.
    cnp = cnp_cost_model()
    cnp_rows: list[dict[str, Any]] = []
    cnp_scores: dict[str, npt.NDArray[np.float64]] = {}
    for strategy in STRATEGIES:
        cnp_scores[strategy] = scores_under(
            strategy, cnp, test_probability, test_amounts, test_loss, loss_model
        )
        cnp_rows.append(
            {
                "policy": strategy,
                **value_recall_at_flag_rate(
                    test_labels, cnp_scores[strategy], test_amounts, MAX_REVIEW_FLAG_RATE, cnp
                ).to_dict(),
            }
        )

    # The tie rule has to apply to the number that QUALIFIES the headline, not only to the one
    # that flatters it. Without this the report sets a headline of -22.4% with an interval
    # excluding zero beside a bare point estimate of 2.2%, which is selective in the direction
    # of this phase's own claim.
    cnp_deltas: list[dict[str, Any]] = []
    for strategy in STRATEGIES[1:]:
        interval = bootstrap_cost_delta(
            test_labels,
            cnp_scores[strategy],
            cnp_scores["probability"],
            test_amounts,
            cnp,
            flag_rate=MAX_REVIEW_FLAG_RATE,
            seed=RANDOM_SEED,
        )
        cnp_candidate = next(row for row in cnp_rows if row["policy"] == strategy)
        cnp_base = next(row for row in cnp_rows if row["policy"] == "probability")
        point = float(cnp_candidate["cost_per_1000"]) - float(cnp_base["cost_per_1000"])
        tie = interval[0] <= 0.0 <= interval[1]
        cnp_deltas.append(
            {
                "policy": strategy,
                "baseline": "probability",
                "cost_delta_per_1000": round(point, 4),
                "cost_delta_ci95": [round(value, 4) for value in interval],
                "cost_delta_pct": (
                    round(100 * point / float(cnp_base["cost_per_1000"]), 4)
                    if cnp_base["cost_per_1000"]
                    else 0.0
                ),
                "verdict": (
                    "TIE -- the interval includes zero"
                    if tie
                    else ("CHEAPER than the baseline" if point < 0 else "MORE EXPENSIVE")
                ),
            }
        )
        logger.info(
            "CNP regime %s vs probability: %+.2f per 1,000, CI [%+.2f, %+.2f] -- %s",
            strategy,
            point,
            interval[0],
            interval[1],
            "tie" if tie else "excludes zero",
        )

    # --- Calibration diagnostics, so the probabilities the costs rest on are checkable.
    from app.models.meta_learner import calibration_curve_points

    calibration_report = calibration_curve_points(test_labels, test_probability)
    calibration = {
        "bins": list(calibration_report.bins),
        "expected_calibration_error": round(calibration_report.expected_calibration_error, 6),
        "brier_score": round(calibration_report.brier, 6),
        "fitted_on": "V-fit",
        "note": (
            "Every cost figure in this entry is a probability multiplied by an amount, so "
            "calibration error propagates directly into cost. Fitted on V-fit; thresholds were "
            "chosen on V-late; this measurement is on test."
        ),
    }

    median_amount = float(np.median(test_amounts))
    default_varying_share = median_amount / (median_amount + cost_model.chargeback_fee)
    cnp_varying_share = median_amount / (median_amount + cnp.chargeback_fee)

    breakeven = amount_aware_threshold(test_amounts, cost_model)
    curve_path = None
    if report_path is not None:
        curve_path = report_path.parent / "cost_curve.png"
        plot_cost_curve(curves, curve_path, cost_model)

    winner = next(r for r in results if r.strategy == headline)
    # At the SHIPPED threshold, because that is what the report says these are. Previously
    # taken at the matched-flag-rate cut, which made the attribution false even though the rows
    # were nearly identical (the matched cut flags fewer rows, so its false-negative set is a
    # superset of the shipped one).
    audit_rows = audit_extremes(
        test,
        test_labels,
        test_scores[headline] >= winner.policy.threshold,
        test_amounts,
        test_probability,
    )

    model_id = build_model_id("causal-cost", "dr-plugin", "ieee-cis")
    return CostRunReport(
        model_id=model_id,
        cost_model=cost_model,
        policies=results,
        deltas=deltas,
        ope=ope,
        calibration=calibration,
        sensitivity=sensitivity,
        sensitivity_rows=sensitivity_rows,
        cnp_regime=cnp_rows,
        cnp_deltas=cnp_deltas,
        median_amount=median_amount,
        slices=slices.describe(),
        audit_rows=audit_rows,
        curve_path=curve_path,
        tier1_model_id=registered.model_id,
        feature_version=registered.model.feature_version,
        training_window={
            "start": str(train["event_time"].min()),
            "end": str(train["event_time"].max()),
        },
        loss_model=loss_model,
        notes=[
            (
                "NO TREATMENT VARIABLE EXISTS IN THIS DATA. IEEE-CIS records no action, "
                "decision, decline, review or dispute column; PaySim's isFlaggedFraud is a "
                "deterministic simulator rule firing on 16 of 6.36M rows, nested inside the "
                "label, and the pipeline drops it. Every transaction here was allowed. No "
                "causal effect is recovered from this corpus and none is claimed."
            ),
            (
                "Under the stated cost model both potential outcomes are deterministic given "
                "(label, amount), so the DR-Learner collapses exactly onto a cost-weighted "
                "plug-in rule driven by a calibrated probability. The proof is in the "
                "app.models.causal_cost module docstring. The DR machinery is used for "
                "off-policy EVALUATION against a SIMULATED logging policy, where the true "
                "policy cost is exactly computable and the estimators can therefore be "
                "validated rather than merely applied."
            ),
            (
                f"Break-even probability is per-transaction, not global: p > r/(A+f+r). Across "
                f"the test split it spans {breakeven.min():.6f} to {breakeven.max():.6f} "
                f"(median {float(np.median(breakeven)):.6f}). That spread is what cost-aware "
                "ranking exploits and a single global threshold cannot."
            ),
            (
                f"Loss regression: LightGBM Tweedie (variance power "
                f"{loss_model.booster.params.get('tweedie_variance_power', 1.5)}) on the target "
                f"Y*(amount+{cost_model.chargeback_fee:.2f}), the same feature set as Tier-1 "
                f"({len(feature_names)} columns). Out-of-fold over {LOSS_OOF_BLOCKS} "
                f"forward-chaining blocks; block 1 unscored, {int(scored.sum()):,} of "
                f"{len(train):,} train rows carry an out-of-fold prediction."
            ),
            (
                "Amount reaches every model in this phase as the engineered amount_log, which "
                "is a live Tier-1 feature. The raw amount and TransactionAmt columns are on the "
                "deny list only because they are exact duplicates of it, monotone and therefore "
                "giving a tree identical splits -- not because amount is withheld. This is not "
                "leakage: the amount is known before the decision is made, which is what makes "
                "it usable in the cost arithmetic at all. The loss regression reads exactly the "
                "columns Tier-1 reads and adds nothing."
            ),
            (
                "The per-transaction worked examples behind the 'what this does NOT catch' "
                "section are deliberately NOT recorded here. This file is tracked in git, and "
                "ten identified transactions with their exact fraud probabilities and the "
                "features that produced them is a worked set of examples of what does and does "
                "not clear the operating threshold. The aggregate failure modes are described "
                "in notebooks/cost_report.md and app/models/README.md, which is what "
                "ml-evaluation-standards section 4 requires; the identifiers add attacker value "
                "and are required by nothing."
            ),
            (
                "THE HEADLINE IS REGIME-DEPENDENT. A false negative costs amount + fee, so what "
                "value-weighting exploits is the share of that cost which varies between "
                f"transactions: {100 * default_varying_share:.1f}% at the default fee of "
                f"{cost_model.chargeback_fee:,.2f} against a median test amount of "
                f"{median_amount:,.2f}, but only {100 * cnp_varying_share:.1f}% at the "
                f"card-not-present fee of {cnp.chargeback_fee:,.2f}. Under that regime the "
                "advantage largely disappears, as the cnp_regime block records. Cost-aware "
                "ranking pays in proportion to how heterogeneous the loss is, and that is a "
                "property of a business's chargeback economics rather than of this model."
            ),
            (
                f"Test was scored once, after all operating points were fixed on V-late. The "
                f"cost matrix is the project default (review {cost_model.review_cost:.2f}, fee "
                f"{cost_model.chargeback_fee:.2f}) so these figures are comparable with Phases "
                "2-5; the card-not-present regime (50/500) is reported as a separate table and "
                "selects nothing."
            ),
        ],
    )


# ==========================================================================================
# Reporting
# ==========================================================================================


def _policy_table(report: CostRunReport) -> list[str]:
    """Return the matched-flag-rate comparison table."""
    lines = [
        "| policy | threshold | flag rate | precision | recall (count) | recall (value) "
        "| TN | FP | FN | TP | cost / 1,000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in report.policies:
        matched = policy.matched
        counts = matched.confusion
        lines.append(
            f"| `{policy.strategy}` | {format_threshold(matched.threshold)} | "
            f"{100 * matched.flag_rate:.3f}% | "
            f"{counts.precision:.4f} | {counts.recall:.4f} | "
            f"**{100 * matched.recall_by_value:.2f}%** | "
            f"{counts.true_negatives:,} | {counts.false_positives:,} | "
            f"{counts.false_negatives:,} | {counts.true_positives:,} | "
            f"{matched.estimate.cost_per_1000_units:,.2f} |"
        )
    return lines


def _delta_table(report: CostRunReport) -> list[str]:
    """Return the paired-interval table -- this phase's headline."""
    lines = [
        "| comparison | cost delta / 1,000 | 95% CI | value recall delta | 95% CI | verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for delta in report.deltas:
        cost_ci = delta["cost_delta_ci95"]
        value_ci = delta["value_recall_delta_ci95"]
        lines.append(
            f"| `{delta['policy']}` vs `{delta['baseline']}` | "
            f"{delta['cost_delta_per_1000']:+,.2f} "
            f"({delta['cost_delta_pct']:+.2f}%) | "
            f"[{cost_ci[0]:+,.2f}, {cost_ci[1]:+,.2f}] | "
            f"{100 * delta['value_recall_delta']:+.2f}pp | "
            f"[{100 * value_ci[0]:+.2f}, {100 * value_ci[1]:+.2f}] | "
            f"{delta['verdict']} |"
        )
    return lines


def render_report(report: CostRunReport, sample: int | None) -> str:
    """Render the Phase 6 markdown report."""
    cost_model = report.cost_model
    lines = [
        "# Phase 6 — Causal Cost Layer",
        "",
        f"Random seed {RANDOM_SEED}. Model `{report.model_id}`, consuming Tier-1 "
        f"`{report.tier1_model_id}` (feature version `{report.feature_version}`).",
    ]
    if sample is not None:
        lines += [
            "",
            f"> **NOT REPORTABLE.** Run with `--sample {sample}`, so every split is truncated "
            "to its earliest rows and no figure below is a held-out result.",
        ]

    lines += [
        "",
        "## There is no treatment variable in this data",
        "",
        "The phase brief asks for inverse probability weighting on the historical actions. "
        "**Neither corpus records an action.** IEEE-CIS carries 394 transaction columns and 41 "
        "identity columns; not one is a decision, decline, review or dispute, and `M1`-`M9` are "
        "Vesta address-match flags rather than review outcomes. PaySim carries one action-like "
        "column, `isFlaggedFraud`, and it fails three ways at once: it is the simulator's own "
        "hardcoded rule, so its propensity is exactly 0 or 1 and `1/e(x)` is undefined; it is "
        "nested inside the label, so the treated arm has no control counterfactual; and it fires "
        "on 16 rows out of 6.36 million. The pipeline already drops it as a leaked downstream "
        "decision.",
        "",
        "Every transaction in this project's data was allowed. **No causal effect is recovered "
        "here and none is claimed.**",
        "",
        "### What follows from that, proved rather than asserted",
        "",
        "Under the stated cost model both potential outcomes are deterministic given the label "
        "and the amount:",
        "",
        "```",
        "cost(block | Y) = (1 - Y) * r          a review is paid only when the row was legitimate",
        "cost(allow | Y) = Y * (A + f)          a missed fraud loses the amount, plus the fee",
        "",
        "E[cost(block) | x] = (1 - p(x)) * r",
        "E[cost(allow) | x] = p(x) * (A + f)",
        "",
        "tau(x) = (1 - p(x)) * r  -  p(x) * (A + f)",
        "```",
        "",
        "Every term is either known before the decision or is `p(x)`. There is no residual "
        "confounding for a doubly robust correction to remove, because there is no treatment "
        "whose assignment could be confounded. **The DR-Learner does not fail here — it "
        "collapses, exactly, onto a cost-weighted plug-in rule.** Two things follow, and this "
        "phase builds both.",
        "",
        "**The decision threshold stops being global.** Blocking pays when "
        "`p(x) > r / (A + f + r)`, a cut-off that moves with the amount. "
        + next(note for note in report.notes if note.startswith("Break-even")),
        "",
        "**There is still something to learn.** The plug-in reaches expected loss through "
        "`p(x) * (A + f)`, a classifier fitted over counts. The `learned_loss` policy regresses "
        "the realised loss directly, weighting large frauds during fitting rather than only at "
        "decision time. Whether that helps is measured below, not assumed.",
        "",
        "## Held-out test",
        "",
        f"All three policies rank the same calibrated Tier-1 probability and are held to the "
        f"same {MAX_REVIEW_FLAG_RATE:.0%} review-capacity cap, chosen on V-late. Test was scored "
        "once, after every operating point was fixed, and selects nothing.",
        "",
        "**Two operating points appear in this report and they are not the same number.** The "
        "blocks below are the *shipped* point: a threshold chosen on V-late and transferred to "
        "test unchanged, which is the honest measurement because nothing on test was used to "
        f"pick it. Its realised flag rate is therefore near {MAX_REVIEW_FLAG_RATE:.0%} but not "
        "exactly it, because test's score distribution is not V-late's. The table that follows "
        "matches all three policies to the same flag rate *as a quantile of the test scores*, "
        "which is what makes their precision figures comparable with each other — and is why "
        "that table selects nothing and is reported only after the shipped thresholds were "
        "fixed. Confusion matrix and cost block within any one result are always the same cut.",
        "",
    ]

    for policy in report.policies:
        lines += ["```", policy.result.render(), "```", ""]

    lines += [
        "### All three at a matched 1% flag rate",
        "",
        *_policy_table(report),
        "",
        "### The headline: paired intervals on the cost difference",
        "",
        "Pre-registered before test was read: **if the interval straddles zero, the comparison "
        "is a tie**, however far apart the point estimates look. Negative cost delta means "
        "cheaper than the named baseline.",
        "",
        "Two questions, three rows. The first two ask whether cost-aware ranking beats "
        "probability ranking at all. The third asks the question BUILD_LOG actually handed this "
        "phase — whether cost-sensitive *training* buys anything over cost-sensitive "
        "*thresholding* — and it is measured directly rather than inferred from the first two. "
        "Two intervals that share a baseline and overlap say nothing about how the candidates "
        "compare with each other, so `learned_loss` is paired against `plug_in` on its own.",
        "",
        *_delta_table(report),
        "",
        "The bootstrap is stratified and paired: both policies are scored on the same resample, "
        "and the threshold is re-derived inside each one rather than held at the value the full "
        "sample produced. Because the resample holds the fraud count fixed, the interval "
        "describes uncertainty in *which* frauds and therefore in their amounts — and cost on "
        "this corpus is dominated by a handful of large transactions, so the intervals are wide. "
        "That width is the finding, not a defect in the estimator.",
        "",
        "## Cost across the full operating range",
        "",
    ]
    if report.curve_path is not None:
        lines += [
            f"![Cost and value capture across the operating range]({report.curve_path.name})",
            "",
            "Computed on the **held-out test split** (n=88,581). Left: cost per 1,000 against "
            "flag rate. Right: share of fraud *value* captured. The dashed line is the 1% "
            "capacity cap. The two panels are the whole argument of this phase — policies that "
            "rank identically on count can separate on value.",
            "",
        ]

    lines += [
        "## Cost-optimal operating points",
        "",
        "Each policy's unconstrained cost minimum, chosen on V-late and applied to test. "
        "Reported beside the capacity-capped points rather than instead of them: the "
        "unconstrained optimum flags more traffic than any review queue could absorb, which is "
        "the assumption the cost model calls the one that bites hardest.",
        "",
        "| policy | threshold | flag rate | cost / 1,000 | vs capped |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in report.policies:
        optimum, matched = policy.optimum, policy.matched
        lines.append(
            f"| `{policy.strategy}` | {format_threshold(optimum.threshold)} | "
            f"{100 * optimum.flag_rate:.3f}% | "
            f"{optimum.estimate.cost_per_1000_units:,.2f} | "
            f"{optimum.estimate.cost_per_1000_units - matched.estimate.cost_per_1000_units:+,.2f} |"
        )

    late_rows = next(
        (row["rows"] for row in report.slices if row["slice"] == "V-late"), 0
    )
    lines += [
        "",
        "## Sensitivity — how much of this rests on the guessed constants",
        "",
        "```",
        render_sensitivity(
            [row for row in report.sensitivity_rows if row.label == "both costs"],
            f"Cost sensitivity, both parameters scaled together "
            f"[V-LATE VALIDATION SLICE, n={late_rows:,} -- not test]",
        ),
        "```",
        "",
        "```",
        render_sensitivity(
            [row for row in report.sensitivity_rows if row.label == "review cost only"],
            f"Review cost sensitivity, false-positive cost alone "
            f"[V-LATE VALIDATION SLICE, n={late_rows:,} -- not test]",
        ),
        "```",
        "",
        "### The card-not-present regime",
        "",
        f"Published CNP figures price a false positive far above analyst time. Re-derived end to "
        f"end at review cost 50 and chargeback fee 500, rather than rescaled. **This table "
        f"selects nothing** — the project default (review {cost_model.review_cost:.2f}, fee "
        f"{cost_model.chargeback_fee:.2f}) remains the basis for every headline above, so that "
        "these numbers stay comparable with Phases 2-5.",
        "",
        "| policy | flag rate | recall (value) | cost / 1,000 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report.cnp_regime:
        lines.append(
            f"| `{row['policy']}` | {100 * row['flag_rate']:.3f}% | "
            f"{100 * row['recall_by_value']:.2f}% | {row['cost_per_1000']:,.2f} |"
        )

    default_delta = next(
        row
        for row in report.deltas
        if row["policy"] == "plug_in" and row["baseline"] == "probability"
    )
    cnp = cnp_cost_model()
    default_varying = report.median_amount / (report.median_amount + cost_model.chargeback_fee)
    cnp_varying = report.median_amount / (report.median_amount + cnp.chargeback_fee)

    cnp_plug_in = next(row for row in report.cnp_deltas if row["policy"] == "plug_in")
    lines += [
        "",
        "| comparison (CNP regime) | cost delta / 1,000 | 95% CI | verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    for delta in report.cnp_deltas:
        interval = delta["cost_delta_ci95"]
        lines.append(
            f"| `{delta['policy']}` vs `{delta['baseline']}` | "
            f"{delta['cost_delta_per_1000']:+,.2f} ({delta['cost_delta_pct']:+.2f}%) | "
            f"[{interval[0]:+,.2f}, {interval[1]:+,.2f}] | {delta['verdict']} |"
        )

    lines += [
        "",
        "**This is the most important caveat on the headline, and it is not a small one.** "
        f"Cost-aware ranking saves {-default_delta['cost_delta_pct']:.1f}% under the project "
        f"cost model. Under the card-not-present one the point estimate is "
        f"{-cnp_plug_in['cost_delta_pct']:.1f}% and **the interval is the verdict above** — the "
        "same pre-registered tie rule that governs the headline governs the number that "
        "qualifies it, because applying it only to the figure that flatters the phase would be "
        "selective. The "
        "mechanism is arithmetic, not noise. A false negative costs `amount + fee`, so the "
        "*share of that cost which varies between transactions* is what value-weighting has to "
        f"work with. At the default fee of {cost_model.chargeback_fee:,.2f} against a median "
        f"test amount of {report.median_amount:,.2f}, that varying share is "
        f"{100 * default_varying:.1f}%. At the card-not-present fee of "
        f"{cnp.chargeback_fee:,.2f} it falls to {100 * cnp_varying:.1f}% — the flat fee "
        "dominates, every miss costs roughly the same, and ranking by expected cost collapses "
        "back towards ranking by probability.",
        "",
        "So the honest statement of this phase's result is conditional: **cost-aware ranking "
        "pays in proportion to how much of the loss varies across transactions.** Where a "
        "missed fraud costs mostly the transaction value, the gain is large. Where it costs "
        "mostly a fixed dispute-handling fee, there is little to exploit and the extra "
        "machinery is not worth its complexity. Which regime a given payments business is in "
        "is an empirical question about its own chargeback economics, not something this "
        "corpus can answer.",
        "",
        "## Off-policy evaluation: validating the estimator, not measuring an effect",
        "",
        "There is no logged policy to reweight, so one is **simulated** with a propensity known "
        "by construction. That is what makes this section worth running: because both potential "
        "outcomes are deterministic given the label and the amount, the true cost of each policy "
        "is *computable exactly*, and the estimators can be scored against it rather than "
        "trusted. A deployment with real logged decisions needs exactly this machinery; here it "
        "can be checked.",
        "",
        "One thing to read carefully. The propensity here is **known exactly**, because this "
        "module wrote it. Under a correct propensity IPW is unbiased by construction, so it is "
        "not a surprise or an achievement when it lands close to the truth — and the doubly "
        "robust estimator, whose correction term adds variance it does not need in that "
        "setting, can legitimately land further away. What this section establishes is that "
        "the estimators are implemented correctly and recover a known answer. It cannot "
        "establish which would win where the propensity has to be estimated, which is the case "
        "that actually matters in deployment and is not testable on data with no logged "
        "actions at all.",
        "",
    ]
    for entry in report.ope:
        lines += ["```", entry.render(), "```", ""]

    lines += [
        "## Calibration",
        "",
        f"Expected calibration error {report.calibration['expected_calibration_error']:.6f}, "
        f"Brier score {report.calibration['brier_score']:.6f}. Every cost figure in this report "
        "is a probability multiplied by an amount, so calibration error propagates straight "
        "into cost — a 10% overstatement of `p` overstates every expected loss by 10% and moves "
        "every break-even threshold with it. The calibrator was fitted on V-fit; thresholds "
        "were chosen on V-late; this measurement is on test.",
        "",
        "## Audit: the costliest mistakes",
        "",
        "Required by ml-evaluation-standards section 4 — the failure modes below are read off "
        "actual false negatives, not imagined.",
        "",
        "### Five costliest false negatives (fraud that got through)",
        "",
        "| amount | p(fraud) | product | new device | new address | prior txns |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in report.audit_rows["false_negatives"]:
        lines.append(
            f"| {row['amount']:,.2f} | {row['fraud_probability']:.6f} | {row['product']} | "
            f"{row['device_is_new']} | {row['addr_is_new']} | "
            f"{row['account_prior_txn_count']} |"
        )
    lines += [
        "",
        "### Five costliest false positives (legitimate customers declined)",
        "",
        "| amount | p(fraud) | product | new device | new address | prior txns |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in report.audit_rows["false_positives"]:
        lines.append(
            f"| {row['amount']:,.2f} | {row['fraud_probability']:.6f} | {row['product']} | "
            f"{row['device_is_new']} | {row['addr_is_new']} | "
            f"{row['account_prior_txn_count']} |"
        )

    lines += [
        "",
        "## Validation, cut three ways",
        "",
        "| slice | rows | positives | base rate | window |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.slices:
        lines.append(
            f"| {row['slice']} | {row['rows']:,} | {row['positives']:,} | "
            f"{100 * row['base_rate']:.4f}% | {row['first_event']} → {row['last_event']} |"
        )

    lines += [
        "",
        "## What this layer does NOT do",
        "",
        "- **It does not recover a causal effect.** There is no treatment variable in this data. "
        "The off-policy section validates estimators against a simulated policy; it measures "
        "nothing about what a real reviewer would have done.",
        "- **It does not know the true cost of a false positive.** The review cost is a stated "
        "assumption, and the sensitivity tables above exist because the recommendation moves "
        "with it.",
        "- **It does not improve detection.** Every policy reads the same Tier-1 probability. "
        "Cost-aware ranking changes which of the flagged transactions are worth the queue slot; "
        "it cannot find fraud Tier-1 did not score highly in the first place.",
        "- **Its advantage shrinks when the chargeback fee dominates the amount.** The gain "
        "comes entirely from heterogeneity in what a miss costs. Price a false negative as a "
        "large flat fee plus a small amount and there is almost nothing left to exploit; the "
        "card-not-present table above shows exactly that.",
        "- **It assumes review capacity is a hard cap and cost is linear in mistakes.** No "
        "queue-congestion effect, no customer-churn effect from repeated false declines, and "
        "no recovery on disputed fraud. All three would raise the true cost of a false positive.",
        "- **It prices `review` and `block` identically.** Both put a transaction in front of a "
        "human and neither lets it complete, but a hard decline damages a customer relationship "
        "in a way a review does not, and this model cannot see the difference.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


# ==========================================================================================
# Registry
# ==========================================================================================


def register(
    report: CostRunReport,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> RegistryEntry:
    """Append this run to the model registry, after writing its artefacts."""
    headline = report.by_strategy("plug_in")
    baseline = report.by_strategy("probability")
    artifact = headline.policy.save(report.model_id, artifact_dir)
    # The shipped plug_in policy holds no booster, so without this the Tweedie model behind every
    # learned_loss figure in this entry would never reach disk and those numbers would not be
    # reconstructable -- doubly so given the row-order-dependent bagging noted in LOSS_PARAMS.
    learned = report.by_strategy("learned_loss")
    learned.policy.save(f"{report.model_id}-loss", artifact_dir)
    delta = next(
        row
        for row in report.deltas
        if row["policy"] == "plug_in" and row["baseline"] == "probability"
    )

    entry = RegistryEntry(
        model_id=report.model_id,
        layer="causal_cost",
        algorithm="cost_weighted_plugin + tweedie_loss_regression",
        source_dataset="ieee_cis",
        feature_version=report.feature_version,
        training_window=report.training_window,
        hyperparameters={
            "consumes_tier1": report.tier1_model_id,
            "review_cost": report.cost_model.review_cost,
            "chargeback_fee": report.cost_model.chargeback_fee,
            "review_capacity_cap": MAX_REVIEW_FLAG_RATE,
            "loss_model": report.loss_model.to_dict(),
            "optimal_threshold": headline.optimum.threshold,
            "shipped_threshold": headline.policy.threshold,
        },
        random_seed=RANDOM_SEED,
        heldout_test={
            **headline.result.to_dict(),
            "unit_of_analysis": "decision",
            "policies": [policy.to_dict() for policy in report.policies],
            "cost_reduction_vs_baseline": delta,
            "cost_per_transaction_at_shipped_threshold": round(
                headline.shipped.estimate.cost_per_1000_units / 1_000, 6
            ),
            "cost_per_transaction_at_matched_flag_rate": round(
                headline.matched.estimate.cost_per_1000_units / 1_000, 6
            ),
            "confusion_matrix_at_optimum": headline.optimum.confusion.to_dict(),
            "ope_validation": [entry.to_dict() for entry in report.ope],
            "calibration": report.calibration,
            "sensitivity": report.sensitivity,
            "cnp_regime": report.cnp_regime,
            "cnp_regime_deltas": report.cnp_deltas,
            "validation_slices": report.slices,
        },
        baseline_comparison=[
            {
                "configuration": "probability ranking (Tier-1 baseline, Phase 2 behaviour)",
                "cost_per_1000": round(baseline.matched.estimate.cost_per_1000_units, 4),
                "recall_by_value": round(baseline.matched.recall_by_value, 6),
                "recall_by_count": round(baseline.matched.confusion.recall, 6),
            },
            *[
                {
                    "configuration": f"{policy.strategy} ranking",
                    "cost_per_1000": round(policy.matched.estimate.cost_per_1000_units, 4),
                    "recall_by_value": round(policy.matched.recall_by_value, 6),
                    "recall_by_count": round(policy.matched.confusion.recall, 6),
                }
                for policy in report.policies
                if policy.strategy != "probability"
            ],
        ],
        artifact=str(artifact.relative_to(artifact_dir.parent)).replace("\\", "/"),
        notes=report.notes,
    )
    append_entry(entry, registry_path)
    logger.info("registered %s", report.model_id)
    return entry


# ==========================================================================================
# CLI
# ==========================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 6 end to end."""
    parser = argparse.ArgumentParser(
        prog="python -m app.models.train_cost_learner",
        description="Train and register the RiskIQ causal cost layer (Phase 6).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the data directory.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Earliest N rows per split. For iteration only; results are not reportable, and the "
            "cost figures degrade badly because cost is dominated by rare large amounts."
        ),
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Write no artefacts and append no registry entry.",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="Where to write the markdown report."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    settings = get_settings()
    if args.data_dir is not None:
        settings = settings.model_copy(update={"data_dir": args.data_dir})
    report_path = args.report or (settings.reports_dir / "cost_report.md")

    try:
        report = run(settings, sample=args.sample, report_path=report_path)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(report, args.sample), encoding="utf-8")
    logger.info("wrote %s", report_path)

    if not args.skip_registry:
        register(report)
    else:
        logger.info("--skip-registry: no artefacts written, no registry entry appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
