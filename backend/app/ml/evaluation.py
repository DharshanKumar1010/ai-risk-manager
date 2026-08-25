"""Held-out evaluation: PR-AUC, confusion matrix, and how confident the difference is.

Three choices here are load-bearing rather than stylistic, and each is a rule from
``.claude/skills/ml-evaluation-standards/SKILL.md``:

**PR-AUC, not ROC-AUC.** At a 3.5% (IEEE-CIS) or 0.97% (PaySim test) positive rate, ROC-AUC
is dominated by the vast negative class and reads impressively for a model that is unusable.
Average precision does not flatter imbalance.

**The no-skill floor is quoted with every PR-AUC.** A random ranker's average precision equals
the positive base rate, so "PR-AUC 0.28" means nothing until you know whether the floor is
0.0097 or 0.35. :func:`no_skill_pr_auc` computes it and :class:`EvaluationResult` carries it,
so the two can never be reported apart.

**Differences get an interval, not a point.** The IEEE-CIS test split holds 3,083 positives.
A PR-AUC gap of a point or two between two models is inside the noise, and declaring a winner
on the point estimate alone is exactly the overclaim the honesty rules exist to prevent —
so :func:`bootstrap_pr_auc` and :func:`bootstrap_pr_auc_delta` resample the test split and
report a percentile interval.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, precision_recall_curve

from app.ml.cost import CostEstimate, CostModel, cost_at_threshold, threshold_for_flag_rate


def format_threshold(threshold: float) -> str:
    """Render a threshold so that distinct values print distinctly and faithfully.

    ``%.6f`` was the original format and it had two failure modes, both observed. Phase 5
    shipped a review/block band 0.00175 wide and printed both ends identically; Tier-3's
    operating threshold of 0.999995 printed as ``1.000000``. A threshold that cannot be read
    back is not a reported operating point, and ml-evaluation-standards section 2 requires the
    operating threshold to be stated.

    Fixed notation is kept whenever it round-trips **exactly**, because that is what a reader
    expects and it covers most thresholds in this project. Anything else falls back to
    :func:`repr`, which is the shortest representation that reads back as the same double.

    Scientific notation was tried here first and was worse, not better: ``%.6e`` carries six
    digits after the point regardless of magnitude, so for a threshold of 90.85343882831513 it
    loses more precision than the ``%.6f`` it was replacing (1.2e-6 against 1.7e-7). Six
    significant digits is not enough for a cost-scale threshold, and the fix for "not enough
    digits" is more digits rather than a different exponent.
    """
    if threshold == float("inf"):
        return "never flag"
    if threshold == float("-inf"):
        return "flag everything"
    fixed = f"{threshold:.6f}"
    return fixed if float(fixed) == threshold else repr(threshold)


#: Resamples for a bootstrap interval. 1,000 is enough for a stable 95% percentile interval
#: and cheap even at 88k test rows, because each resample only re-sorts existing scores.
BOOTSTRAP_RESAMPLES = 1_000

#: Two-sided interval width. Reported as a 95% percentile interval.
BOOTSTRAP_ALPHA = 0.05

#: Above this, a fraud-detection PR-AUC is treated as a suspected leak rather than a result —
#: ml-evaluation-standards section 4. Held here so every tier trips the same wire.
LEAK_SUSPICION_PR_AUC = 0.95


@dataclass(frozen=True)
class ConfusionMatrix:
    """Raw counts at one threshold. Never percentages — percentages hide the denominator."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total(self) -> int:
        """Return the number of rows scored."""
        return (
            self.true_negatives + self.false_positives + self.false_negatives + self.true_positives
        )

    @property
    def precision(self) -> float:
        """Return TP / (TP + FP), or 0.0 when nothing was flagged."""
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Return TP / (TP + FN), or 0.0 when there are no positives."""
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives else 0.0

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, int]:
        """Return the registry/JSON shape."""
        return {
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tp": self.true_positives,
        }


def confusion_at_threshold(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    threshold: float,
) -> ConfusionMatrix:
    """Return the confusion matrix for ``scores >= threshold``.

    Args:
        labels: True where the row is fraud.
        scores: Higher means more suspicious. Every model in this project scores in that
            direction; an Isolation Forest's ``score_samples`` must be negated before it
            reaches here.
        threshold: Flag at or above this score.
    """
    predicted = scores >= threshold
    return ConfusionMatrix(
        true_negatives=int(np.sum(~labels & ~predicted)),
        false_positives=int(np.sum(~labels & predicted)),
        false_negatives=int(np.sum(labels & ~predicted)),
        true_positives=int(np.sum(labels & predicted)),
    )


def no_skill_pr_auc(labels: npt.NDArray[np.bool_]) -> float:
    """Return the PR-AUC a random ranker achieves on this label vector.

    That is simply the positive base rate. It is the floor every reported PR-AUC must clear
    to mean anything, which is why it is computed rather than assumed.
    """
    return float(np.mean(labels)) if labels.size else 0.0


def pr_auc(labels: npt.NDArray[np.bool_], scores: npt.NDArray[np.float64]) -> float:
    """Return average precision — the headline metric for every model in this project."""
    if labels.size == 0 or not labels.any():
        return 0.0
    return float(average_precision_score(labels, scores))


def _percentile_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return the two-sided percentile interval at :data:`BOOTSTRAP_ALPHA`."""
    low, high = 100 * BOOTSTRAP_ALPHA / 2, 100 * (1 - BOOTSTRAP_ALPHA / 2)
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def _bootstrap_indices(
    labels: npt.NDArray[np.bool_],
    resamples: int,
    seed: int,
) -> Sequence[npt.NDArray[np.intp]]:
    """Yield stratified bootstrap index arrays, preserving the positive count exactly.

    Stratified rather than plain: at a 0.97% base rate an unstratified resample occasionally
    draws very few positives, which inflates the variance of the interval for a reason that
    has nothing to do with the models being compared.
    """
    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(labels)
    negative_index = np.flatnonzero(~labels)
    return [
        np.concatenate(
            (
                rng.choice(positive_index, size=positive_index.size, replace=True),
                rng.choice(negative_index, size=negative_index.size, replace=True),
            )
        )
        for _ in range(resamples)
    ]


def bootstrap_pr_auc(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile confidence interval for one model's PR-AUC.

    Args:
        labels: True where the row is fraud.
        scores: Higher means more suspicious.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling, logged with the result.

    Returns:
        ``(low, high)`` at :data:`BOOTSTRAP_ALPHA`.
    """
    if not labels.any():
        return 0.0, 0.0
    draws = np.array(
        [
            pr_auc(labels[index], scores[index])
            for index in _bootstrap_indices(labels, resamples, seed)
        ]
    )
    return _percentile_interval(draws)


def bootstrap_pr_auc_delta(
    labels: npt.NDArray[np.bool_],
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a confidence interval for ``PR-AUC(a) - PR-AUC(b)``.

    Both models are scored on the *same* resample, which is the point: it pairs them, so the
    interval reflects the difference between the models rather than the variance of the test
    split itself. An interval that straddles zero means the comparison is a tie, however
    different the point estimates look.
    """
    if not labels.any():
        return 0.0, 0.0
    draws = np.array(
        [
            pr_auc(labels[index], scores_a[index]) - pr_auc(labels[index], scores_b[index])
            for index in _bootstrap_indices(labels, resamples, seed)
        ]
    )
    return _percentile_interval(draws)


@dataclass(frozen=True)
class EvaluationResult:
    """One model's measured performance on one split.

    Deliberately bundles the metric with the things that make it interpretable — the split it
    came from, its base rate, the threshold, and the no-skill floor — so a caller cannot
    report the headline without them.
    """

    model_name: str
    split: str
    threshold: float
    threshold_criterion: str
    rows: int
    positives: int
    pr_auc: float
    pr_auc_interval: tuple[float, float] | None
    confusion: ConfusionMatrix
    false_positive_cost: "CostEstimate | None" = None
    notes: list[str] = field(default_factory=list)

    @property
    def base_rate(self) -> float:
        """Return the positive-class rate on this split."""
        return self.positives / self.rows if self.rows else 0.0

    @property
    def lift_over_no_skill(self) -> float:
        """Return PR-AUC as a multiple of the random-ranker floor."""
        return self.pr_auc / self.base_rate if self.base_rate else 0.0

    @property
    def is_leak_suspicious(self) -> bool:
        """Return whether this result should be treated as a leak until disproven."""
        return self.pr_auc > LEAK_SUSPICION_PR_AUC

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape for ``models/registry.json``."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "split": self.split,
            "rows": self.rows,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 6),
            "threshold": self.threshold,
            "threshold_criterion": self.threshold_criterion,
            "pr_auc": round(self.pr_auc, 6),
            "pr_auc_no_skill_floor": round(self.base_rate, 6),
            "lift_over_no_skill": round(self.lift_over_no_skill, 3),
            "precision": round(self.confusion.precision, 6),
            "recall": round(self.confusion.recall, 6),
            "f1": round(self.confusion.f1, 6),
            "confusion_matrix": self.confusion.to_dict(),
        }
        if self.pr_auc_interval is not None:
            payload["pr_auc_ci95"] = [round(value, 6) for value in self.pr_auc_interval]
        if self.false_positive_cost is not None:
            payload["false_positive_cost"] = self.false_positive_cost.to_dict()
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def render(self) -> str:
        """Return the exact reporting block ml-evaluation-standards prescribes."""
        matrix = self.confusion
        interval = (
            f"  (95% CI {self.pr_auc_interval[0]:.4f}-{self.pr_auc_interval[1]:.4f})"
            if self.pr_auc_interval is not None
            else ""
        )
        lines = [
            f"### {self.model_name} — held-out {self.split} "
            f"(n={self.rows:,}, positives={self.positives:,}, "
            f"base rate={100 * self.base_rate:.4f}%)",
            f"Threshold: {format_threshold(self.threshold)}  ({self.threshold_criterion})",
            "",
            f"PR-AUC     {self.pr_auc:.4f}{interval}",
            f"           no-skill floor {self.base_rate:.4f} "
            f"({self.lift_over_no_skill:.1f}x lift)",
            f"Precision  {matrix.precision:.4f}      "
            f"Recall  {matrix.recall:.4f}      F1  {matrix.f1:.4f}",
            "",
            "Confusion matrix        Predicted",
            "                     neg        pos",
            f"Actual  neg  {matrix.true_negatives:>9,}  {matrix.false_positives:>9,}",
            f"        pos  {matrix.false_negatives:>9,}  {matrix.true_positives:>9,}",
        ]
        if self.false_positive_cost is not None:
            lines += ["", self.false_positive_cost.render()]
        if self.is_leak_suspicious:
            lines += [
                "",
                f"!! PR-AUC exceeds {LEAK_SUSPICION_PR_AUC} on fraud data. Treat as a "
                "suspected leak until disproven (ml-evaluation-standards section 4).",
            ]
        for note in self.notes:
            lines += ["", note]
        return "\n".join(lines)


def evaluate(
    model_name: str,
    split: str,
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    *,
    threshold: float,
    threshold_criterion: str,
    interval: tuple[float, float] | None = None,
    cost: "CostEstimate | None" = None,
    notes: Sequence[str] = (),
) -> EvaluationResult:
    """Measure one model on one split and bundle everything needed to report it honestly."""
    return EvaluationResult(
        model_name=model_name,
        split=split,
        threshold=threshold,
        threshold_criterion=threshold_criterion,
        rows=int(labels.size),
        positives=int(np.sum(labels)),
        pr_auc=pr_auc(labels, scores),
        pr_auc_interval=interval,
        confusion=confusion_at_threshold(labels, scores, threshold),
        false_positive_cost=cost,
        notes=list(notes),
    )


def threshold_for_recall(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    target_recall: float,
) -> float:
    """Return the lowest threshold achieving at least ``target_recall``.

    Offered as an alternative operating point for the dashboard, never as the headline
    criterion — ml-evaluation-standards section 3 requires the recommended threshold to be
    justified by cost rather than by a recall target picked in advance.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more precision/recall point than thresholds; the
    # final point is the degenerate (recall=0, precision=1) corner with no threshold.
    usable = np.flatnonzero(recall[:-1] >= target_recall)
    if usable.size == 0:
        return float(np.min(scores))
    return float(thresholds[usable[-1]])


# ==========================================================================================
# Intervals on a cost or value difference
# ==========================================================================================
#
# Phase 5 reported the meta-learner as roughly 1.7% cheaper than Tier-1 at a matched flag rate
# and could not say whether that survived resampling, because this module had an interval for
# a PR-AUC difference and nothing for a cost or value one. BUILD_LOG recorded the gap as a
# known limitation and handed it to Phase 6, whose entire headline is a cost difference.
#
# **One caveat travels with every interval below.** The resample is stratified, so the number
# of frauds is held fixed and only *which* frauds -- and therefore which amounts -- varies.
# For PR-AUC that is a variance reduction with no downside. For cost it means the interval
# describes uncertainty in the amounts, not in the base rate, and cost on this corpus is
# dominated by a handful of large transactions. The intervals are therefore wide, and that
# width is the finding rather than a defect in the estimator.


def _cost_per_1000_at_flag_rate(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    flag_rate: float,
    model: CostModel,
) -> float:
    """Return cost per 1,000 rows at the threshold flagging ``flag_rate`` of them."""
    threshold = threshold_for_flag_rate(scores, flag_rate)
    return cost_at_threshold(labels, scores, amounts, threshold, model).cost_per_1000_units


def bootstrap_cost_delta(
    labels: npt.NDArray[np.bool_],
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
    *,
    flag_rate: float,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Return an interval for ``cost_per_1000(a) - cost_per_1000(b)`` at a matched flag rate.

    Negative means ``a`` is cheaper. An interval straddling zero means the two policies cost
    the same, however far apart their point estimates look.

    The threshold is re-derived inside each resample rather than held fixed at the one the
    full sample produced. It is a quantile of the scores, so it is itself an estimate; holding
    it fixed would report an interval for a decision rule nobody could have chosen without
    having already seen the data.
    """
    if not labels.any():
        return 0.0, 0.0
    draws = np.array(
        [
            _cost_per_1000_at_flag_rate(
                labels[index], scores_a[index], amounts[index], flag_rate, model
            )
            - _cost_per_1000_at_flag_rate(
                labels[index], scores_b[index], amounts[index], flag_rate, model
            )
            for index in _bootstrap_indices(labels, resamples, seed)
        ]
    )
    return _percentile_interval(draws)


def _value_recall_at_flag_rate(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    flag_rate: float,
) -> float:
    """Return the share of fraud value captured at the matched flag rate."""
    total = float(np.sum(amounts[labels]))
    if not total:
        return 0.0
    flagged = scores >= threshold_for_flag_rate(scores, flag_rate)
    return float(np.sum(amounts[labels & flagged])) / total


def bootstrap_value_recall_delta(
    labels: npt.NDArray[np.bool_],
    scores_a: npt.NDArray[np.float64],
    scores_b: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    *,
    flag_rate: float,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Return an interval for the difference in fraud value captured, ``a`` minus ``b``.

    Positive means ``a`` captures more of the money. This is the mechanism behind any cost
    difference between two rankings at the same flag rate, so it is reported beside the cost
    interval rather than inferred from it.
    """
    if not labels.any():
        return 0.0, 0.0
    draws = np.array(
        [
            _value_recall_at_flag_rate(labels[index], scores_a[index], amounts[index], flag_rate)
            - _value_recall_at_flag_rate(labels[index], scores_b[index], amounts[index], flag_rate)
            for index in _bootstrap_indices(labels, resamples, seed)
        ]
    )
    return _percentile_interval(draws)
