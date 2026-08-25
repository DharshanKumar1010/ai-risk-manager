"""False-positive cost — mandatory alongside every reported metric.

``.claude/skills/ml-evaluation-standards/SKILL.md`` section 3 is blunt about this: a metrics
section without a cost estimate is incomplete for this project. The reason is that precision
and recall trade against each other and nothing in either number says which way the trade
should go. Cost does.

The model here is deliberately simple and **constant** — a flat cost per false positive, and
an amount-weighted cost per false negative. Phase 6 replaces it with a DR-Learner estimate of
the *heterogeneous* per-decision cost, which is the track's differentiator; this module is the
honest placeholder that makes Phases 2-5 reportable in the meantime, not an attempt at that.

    total_cost(t) = review_cost * FP(t) + sum over FN(t) of (amount_i + chargeback_fee)

Two things this deliberately does not claim:

**It is an estimate with stated assumptions, never ground truth.** Every number it produces
carries :attr:`CostEstimate.assumptions` and every rendering prints them.

**Costs from different corpora are not comparable and must never be summed.** IEEE-CIS amounts
are currency; PaySim amounts are simulator units whose scale is arbitrary.
:meth:`CostModel.scaled_to` exists precisely so the *ratio* between review cost and typical
loss stays comparable across the two, which is the only thing that transfers.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for typing only
    # app.ml.evaluation imports CostEstimate from here, so the runtime import goes the other
    # way round inside the one function that needs it.
    from app.ml.evaluation import ConfusionMatrix

#: Cost of reviewing or declining one flagged legitimate transaction, in IEEE-CIS amount
#: units (that corpus's amounts are consistent with USD). Covers analyst time on a manual
#: review queue. An assumption, and one of the two parameters the sensitivity sweep varies.
DEFAULT_REVIEW_COST = 3.0

#: Flat fee added to the disputed amount when a fraud succeeds — the issuer/network
#: chargeback fee and internal handling, on top of losing the transaction value itself.
#: Also an assumption, also swept.
DEFAULT_CHARGEBACK_FEE = 15.0

#: Median IEEE-CIS transaction amount, measured in Phase 1 (notebooks/eda_report.md). The
#: reference point :meth:`CostModel.scaled_to` uses to carry this cost structure onto a corpus
#: whose amounts are on a different scale.
IEEE_CIS_MEDIAN_AMOUNT = 68.769

#: Multipliers applied to both cost parameters in the sensitivity analysis. Section 3 asks
#: for +/-50%.
SENSITIVITY_FACTORS: tuple[float, ...] = (0.5, 1.0, 1.5)


@dataclass(frozen=True)
class CostModel:
    """What a mistake costs, in one corpus's own amount units.

    Attributes:
        review_cost: Cost of one false positive.
        chargeback_fee: Flat cost added to the lost amount on one false negative.
        units: Plain-language name for the amount unit, printed with every estimate so two
            corpora's figures are never read as though they were the same currency.
    """

    review_cost: float = DEFAULT_REVIEW_COST
    chargeback_fee: float = DEFAULT_CHARGEBACK_FEE
    units: str = "IEEE-CIS amount units (consistent with USD)"
    #: What one row of the evaluated population *is*. Tiers 1 and 2 score transactions; Tier-3
    #: scores rings, and a ring cost printed "per 1,000 transactions" reads as roughly 60x its
    #: real per-transaction figure. The noun travels with the number.
    unit_noun: str = "transaction"

    @classmethod
    def scaled_to(cls, median_amount: float, units: str) -> "CostModel":
        """Return this cost structure rescaled to a corpus with different-sized amounts.

        PaySim's median transaction is roughly 2,500x IEEE-CIS's. Reusing the flat IEEE-CIS
        review cost there would make reviewing effectively free against the size of a typical
        loss, and the cost-minimising threshold would collapse to "flag everything" — an
        artefact of mismatched units, not a finding about the model.

        Scaling both parameters by the ratio of median amounts keeps the *economics*
        comparable — how many reviews one averted loss pays for — which is the part that
        actually transfers between corpora.

        Args:
            median_amount: Median transaction amount in the target corpus.
            units: Plain-language name for that corpus's amount unit.
        """
        ratio = median_amount / IEEE_CIS_MEDIAN_AMOUNT
        return cls(
            review_cost=DEFAULT_REVIEW_COST * ratio,
            chargeback_fee=DEFAULT_CHARGEBACK_FEE * ratio,
            units=units,
        )

    def with_unit(self, unit_noun: str) -> "CostModel":
        """Return a copy whose cost figures are labelled per ``unit_noun``."""
        return replace(self, unit_noun=unit_noun)

    def scaled_by(self, factor: float) -> "CostModel":
        """Return this model with both cost parameters multiplied by ``factor``."""
        return CostModel(
            review_cost=self.review_cost * factor,
            chargeback_fee=self.chargeback_fee * factor,
            units=self.units,
        )

    def assumptions(self) -> list[str]:
        """Return the plain-language assumption list printed with every estimate."""
        return [
            f"A false positive costs {self.review_cost:,.2f} {self.units} - analyst time on "
            "a manual review, or the lost margin and friction of declining a good customer.",
            f"A false negative costs the transaction amount plus a flat "
            f"{self.chargeback_fee:,.2f} - the chargeback fee and internal handling on top "
            "of the value lost.",
            "Cost is linear in the number of mistakes: no queue-congestion effect, no "
            "customer-churn effect from repeated false declines, no recovery on disputed "
            "fraud. All three would raise the true cost of a false positive.",
            "Every fraud is assumed to charge back. In practice some share is never "
            "disputed, which would lower false-negative cost.",
            "Review capacity is unbounded. This is the assumption that bites hardest: the "
            "cost-minimising threshold below flags whatever share of traffic the arithmetic "
            "favours, with no ceiling on the queue it implies.",
        ]


@dataclass(frozen=True)
class CostEstimate:
    """The cost of one model's decisions at one threshold, with its assumptions attached."""

    threshold: float
    false_positives: int
    false_negatives: int
    false_positive_cost: float
    false_negative_cost: float
    rows: int
    model: CostModel
    flagged: int = 0

    @property
    def total_cost(self) -> float:
        """Return the combined cost of both error types."""
        return self.false_positive_cost + self.false_negative_cost

    @property
    def flag_rate(self) -> float:
        """Return the share of all scored units this threshold sends for review.

        The operationally decisive number, and the one a precision figure hides. Precision
        0.06 at a 40% flag rate and precision 0.06 at a 0.4% flag rate are the same metric
        describing two completely different products — one of which cannot be staffed.
        """
        return self.flagged / self.rows if self.rows else 0.0

    @property
    def cost_per_1000_units(self) -> float:
        """Return cost normalised per 1,000 scored units.

        "Units", not "transactions": Tier-3 evaluates rings, and a ring cost divided by a ring
        count but labelled per-transaction reads as roughly 60x its real per-transaction value.
        The noun comes from :attr:`CostModel.unit_noun` so the label cannot drift from what was
        actually counted.

        The comparable figure across splits and corpora of different sizes — though still
        never across corpora of different *units*.
        """
        return 1_000 * self.total_cost / self.rows if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape."""
        return {
            "threshold": self.threshold,
            "units": self.model.units,
            "review_cost_per_false_positive": round(self.model.review_cost, 4),
            "flat_chargeback_fee_per_false_negative": round(self.model.chargeback_fee, 4),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "false_positive_cost": round(self.false_positive_cost, 2),
            "false_negative_cost": round(self.false_negative_cost, 2),
            "total_cost": round(self.total_cost, 2),
            "unit": self.model.unit_noun,
            f"cost_per_1000_{self.model.unit_noun}s": round(self.cost_per_1000_units, 4),
            "flagged": self.flagged,
            "flag_rate": round(self.flag_rate, 6),
            "assumptions": self.model.assumptions(),
        }

    def render(self) -> str:
        """Return the cost block for the metrics report."""
        lines = [
            f"False-positive cost estimate: {self.cost_per_1000_units:,.2f} "
            f"per 1,000 {self.model.unit_noun}s  [{self.model.units}]",
            f"  {self.false_positives:,} false positives x "
            f"{self.model.review_cost:,.2f} = {self.false_positive_cost:,.2f}",
            f"  {self.false_negatives:,} false negatives (amount + "
            f"{self.model.chargeback_fee:,.2f}) = {self.false_negative_cost:,.2f}",
            f"  total = {self.total_cost:,.2f} over {self.rows:,} {self.model.unit_noun}s",
            f"  flag rate = {100 * self.flag_rate:.2f}% " f"({self.flagged:,} sent for review)",
            "  Assumptions:",
        ]
        lines += [f"    - {assumption}" for assumption in self.model.assumptions()]
        return "\n".join(lines)


def cost_at_threshold(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    threshold: float,
    model: CostModel,
) -> CostEstimate:
    """Return the cost of flagging ``scores >= threshold``."""
    flagged = scores >= threshold
    false_positive = ~labels & flagged
    false_negative = labels & ~flagged
    false_negative_count = int(np.sum(false_negative))
    return CostEstimate(
        threshold=threshold,
        false_positives=int(np.sum(false_positive)),
        false_negatives=false_negative_count,
        false_positive_cost=float(np.sum(false_positive)) * model.review_cost,
        false_negative_cost=float(np.sum(amounts[false_negative]))
        + false_negative_count * model.chargeback_fee,
        rows=int(labels.size),
        model=model,
        flagged=int(np.sum(flagged)),
    )


@dataclass(frozen=True)
class CostCurve:
    """Every achievable operating point, with what each one costs and captures.

    Attributes:
        thresholds: Candidate thresholds, descending in flag rate order.
        costs: Total cost at each threshold.
        caught_value: Fraud value captured at each threshold.
        flagged: Rows flagged at each threshold.
        total_positive_value: Fraud value available to capture. The denominator of value
            recall, carried here so a caller cannot divide by the wrong total.
    """

    thresholds: npt.NDArray[np.float64]
    costs: npt.NDArray[np.float64]
    caught_value: npt.NDArray[np.float64]
    flagged: npt.NDArray[np.float64]
    total_positive_value: float

    @property
    def value_recall(self) -> npt.NDArray[np.float64]:
        """Return the share of fraud value captured at each operating point."""
        if not self.total_positive_value:
            return np.zeros_like(self.caught_value)
        return self.caught_value / self.total_positive_value


def cost_curve(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
) -> CostCurve:
    """Return every achievable operating point, with cost and captured value at each.

    Phase 6 needs the value-capture curve alongside the cost curve, because the two can
    disagree: a model that misses more frauds while missing *cheaper* ones moves down the
    count axis and up the value axis at once. That disagreement is the whole subject of the
    causal cost layer, so both vectors are returned rather than one being recomputed.

    Computed by sorting once and walking cumulative sums rather than by re-scoring the split
    at each candidate: an 88k-row validation split has ~88k distinct scores, and the naive
    sweep is quadratic. This is exact, not an approximation over a grid.

    Only distinct-score boundaries are offered as candidates. A threshold falling between two
    tied scores is not achievable by a ``>=`` rule, and offering it would report a cost the
    deployed decision could never actually realise.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_amounts = amounts[order]

    # Position i means "flag the top i+1 rows".
    true_positives = np.cumsum(sorted_labels)
    flagged = np.arange(1, labels.size + 1)
    false_positives = flagged - true_positives

    total_positives = int(true_positives[-1]) if labels.size else 0
    total_positive_amount = float(np.sum(sorted_amounts[sorted_labels]))
    caught_amount = np.cumsum(np.where(sorted_labels, sorted_amounts, 0.0))

    false_negatives = total_positives - true_positives
    missed_amount = total_positive_amount - caught_amount

    cost = (
        false_positives * model.review_cost + missed_amount + false_negatives * model.chargeback_fee
    )

    # Keep only the last position of each run of tied scores — the achievable boundaries.
    is_boundary = np.ones(labels.size, dtype=bool)
    is_boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    # Prepend the flag-nothing operating point, whose cost is every fraud missed.
    flag_nothing_cost = total_positive_amount + total_positives * model.chargeback_fee
    return CostCurve(
        thresholds=np.concatenate(([np.inf], sorted_scores[is_boundary])),
        costs=np.concatenate(([flag_nothing_cost], cost[is_boundary])),
        caught_value=np.concatenate(([0.0], caught_amount[is_boundary])),
        flagged=np.concatenate(([0.0], flagged[is_boundary].astype(np.float64))),
        total_positive_value=total_positive_amount,
    )


def choose_threshold_by_cost(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
) -> tuple[float, CostEstimate]:
    """Return the cost-minimising threshold and its estimate.

    **Call this on validation only.** Choosing an operating point on test would make test a
    validation set and invalidate the headline, per ml-evaluation-standards section 1.

    Returns:
        ``(threshold, estimate)``. The threshold is ``inf`` when flagging nothing is cheapest,
        which is a real and reportable outcome — it means the model cannot separate well
        enough to pay for its own review queue under these assumptions.
    """
    curve = cost_curve(labels, scores, amounts, model)
    best = int(np.argmin(curve.costs))
    threshold = float(curve.thresholds[best])
    return threshold, cost_at_threshold(labels, scores, amounts, threshold, model)


#: Review costs, as multiples of the default, for the ratio sweep. Scaling both cost
#: parameters together barely moves the recommended threshold; scaling only the review cost
#: moves it a great deal, because it is the FP:FN *ratio* that sets the operating point. This
#: is the sweep that actually tells a reader how much the recommendation rests on a guess.
REVIEW_COST_MULTIPLES: tuple[float, ...] = (1.0, 5.0, 25.0, 100.0)


@dataclass(frozen=True)
class SensitivityRow:
    """One row of a sensitivity analysis."""

    factor: float
    threshold: float
    estimate: CostEstimate
    label: str = "both costs"

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape."""
        return {
            "varied": self.label,
            "factor": self.factor,
            "threshold": self.threshold,
            "total_cost": round(self.estimate.total_cost, 2),
            "false_positives": self.estimate.false_positives,
            "false_negatives": self.estimate.false_negatives,
            "flag_rate": round(self.estimate.flag_rate, 6),
        }


def sensitivity_sweep(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
    factors: Sequence[float] = SENSITIVITY_FACTORS,
) -> list[SensitivityRow]:
    """Re-derive the recommended threshold under scaled cost assumptions.

    Scaling both parameters together is deliberately the *uninformative* direction — it
    changes the magnitude of the cost but not the FP/FN ratio, so a threshold that barely
    moves here is evidence the recommendation is driven by the model's ranking rather than by
    the guessed constants. The informative sweep varies the two independently, which the
    caller can do by passing modified models; this is the section 3 minimum.
    """
    rows: list[SensitivityRow] = []
    for factor in factors:
        scaled = model.scaled_by(factor)
        threshold, estimate = choose_threshold_by_cost(labels, scores, amounts, scaled)
        rows.append(SensitivityRow(factor=factor, threshold=threshold, estimate=estimate))
    return rows


def review_cost_sweep(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
    multiples: Sequence[float] = REVIEW_COST_MULTIPLES,
) -> list[SensitivityRow]:
    """Re-derive the threshold while varying only the false-positive cost.

    The informative direction. A false negative costs the transaction amount — on IEEE-CIS a
    median of roughly 69 units — while a review is assumed to cost 3. At that ~23:1 ratio the
    cost-minimising rule is to review very aggressively, and the resulting flag rate is a
    statement about the assumed ratio at least as much as about the model.

    Raising only the review cost traces how the recommendation moves as false positives are
    priced more like a lost customer than like a few minutes of analyst time, which is the
    range a real deployment sits in.
    """
    rows: list[SensitivityRow] = []
    for multiple in multiples:
        varied = CostModel(
            review_cost=model.review_cost * multiple,
            chargeback_fee=model.chargeback_fee,
            units=model.units,
        )
        threshold, estimate = choose_threshold_by_cost(labels, scores, amounts, varied)
        rows.append(
            SensitivityRow(
                factor=multiple,
                threshold=threshold,
                estimate=estimate,
                label="review cost only",
            )
        )
    return rows


def threshold_for_flag_rate(
    scores: npt.NDArray[np.float64],
    max_flag_rate: float,
) -> float:
    """Return the lowest threshold flagging no more than ``max_flag_rate`` of scored units.

    The capacity-constrained operating point. Every real review queue has a ceiling, and a
    cost-optimal threshold that ignores it is not deployable however good its arithmetic. The
    threshold is derived from the score distribution alone and reads no label, so it may be
    taken from validation or applied directly.
    """
    if scores.size == 0:
        return float("inf")
    quantile = float(np.quantile(scores, 1.0 - max_flag_rate))
    # Nudge above ties so the realised rate does not overshoot when many rows share a score.
    above = scores[scores > quantile]
    return float(np.min(above)) if above.size else float("inf")


def render_sensitivity(rows: Sequence[SensitivityRow], title: str) -> str:
    """Return a sensitivity table for the metrics report."""
    lines = [
        title,
        "  factor   threshold        total cost        FP        FN   flag rate",
    ]
    for row in rows:
        threshold = "never flag" if np.isinf(row.threshold) else f"{row.threshold:.6f}"
        lines.append(
            f"  {row.factor:>5.1f}x  {threshold:>12}  {row.estimate.total_cost:>16,.2f}  "
            f"{row.estimate.false_positives:>8,}  {row.estimate.false_negatives:>8,}  "
            f"{100 * row.estimate.flag_rate:>8.2f}%"
        )
    return "\n".join(lines)


#: The card-not-present regime: a false positive priced as a lost customer rather than as a
#: few minutes of analyst time, and a false negative carrying a dispute-handling cost an order
#: of magnitude above the bare network fee. Published CNP figures cluster here.
#:
#: Reported alongside the project defaults rather than replacing them. Every cost number in
#: Phases 2-5 is on the default basis, and silently moving the basis would make this phase's
#: figures look like an improvement on numbers they are not comparable to.
CNP_REVIEW_COST = 50.0
CNP_CHARGEBACK_FEE = 500.0


def cnp_cost_model() -> CostModel:
    """Return the card-not-present cost regime, for the sensitivity section."""
    return CostModel(review_cost=CNP_REVIEW_COST, chargeback_fee=CNP_CHARGEBACK_FEE)


@dataclass(frozen=True)
class ValueRecall:
    """What one operating point captures, by count and by value.

    The two diverge, and the divergence is the point. A model can miss *more* frauds than a
    rival and still cost less by missing cheaper ones, so a count-recall comparison on its own
    can rank two models in the opposite order to what they cost.

    Attributes:
        threshold: The operating point.
        flag_rate: Share of scored rows flagged.
        caught_value: Fraud value captured.
        total_value: Fraud value available.
        confusion: Counts at this threshold.
        estimate: The cost of this operating point.
    """

    threshold: float
    flag_rate: float
    caught_value: float
    total_value: float
    confusion: "ConfusionMatrix"
    estimate: CostEstimate

    @property
    def recall_by_value(self) -> float:
        """Return the share of fraud *value* captured."""
        return self.caught_value / self.total_value if self.total_value else 0.0

    @property
    def missed_value(self) -> float:
        """Return the fraud value that got through."""
        return self.total_value - self.caught_value

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape."""
        return {
            "threshold": self.threshold,
            "flag_rate": round(self.flag_rate, 6),
            "precision": round(self.confusion.precision, 6),
            "recall": round(self.confusion.recall, 6),
            "recall_by_value": round(self.recall_by_value, 6),
            "caught_fraud_value": round(self.caught_value, 2),
            "missed_fraud_value": round(self.missed_value, 2),
            "confusion_matrix": self.confusion.to_dict(),
            "cost_per_1000": round(self.estimate.cost_per_1000_units, 2),
        }


def value_recall_at_threshold(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    threshold: float,
    model: CostModel,
) -> ValueRecall:
    """Return count recall, value recall and cost at one operating point.

    Lifted out of the Phase 5 driver, where it was a local block inside the run function.
    Phase 6 compares three ranking policies on exactly these quantities, and a metric computed
    twice in two places is a metric that can disagree with itself.
    """
    from app.ml.evaluation import confusion_at_threshold

    flagged = scores >= threshold
    return ValueRecall(
        threshold=threshold,
        flag_rate=float(np.mean(flagged)) if flagged.size else 0.0,
        caught_value=float(np.sum(amounts[labels & flagged])),
        total_value=float(np.sum(amounts[labels])),
        confusion=confusion_at_threshold(labels, scores, threshold),
        estimate=cost_at_threshold(labels, scores, amounts, threshold, model),
    )


def value_recall_at_flag_rate(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    max_flag_rate: float,
    model: CostModel,
) -> ValueRecall:
    """Return the same, at the threshold flagging ``max_flag_rate`` of scored rows.

    The matched-flag-rate view. Two models compared at their own thresholds are two products
    with different staffing costs; matched to the same review capacity, the comparison is
    about ranking quality alone.

    **The threshold is a quantile of the scores it is applied to.** That is what makes the flag
    rates comparable, and it is why a matched-flag-rate table selects nothing and must be
    reported only after the shipped operating points have been fixed on validation.
    """
    threshold = threshold_for_flag_rate(scores, max_flag_rate)
    return value_recall_at_threshold(labels, scores, amounts, threshold, model)
