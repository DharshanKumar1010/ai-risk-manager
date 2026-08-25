"""Off-policy evaluation: what would a different decision policy have cost?

**Read this before reading any number this module produces.**

The Phase 6 brief asks for inverse probability weighting "on the historical actions -- what was
actually done". Neither corpus records an action. IEEE-CIS carries 394 transaction columns and
41 identity columns and not one of them is a decision, a decline, a review or a dispute;
``M1``-``M9`` are Vesta address-match flags, not review outcomes. PaySim carries exactly one
action-like column, ``isFlaggedFraud``, and it is unusable three ways over: it is a hardcoded
rule belonging to the simulator itself (a single TRANSFER above 200,000), so its propensity is
exactly 0 or 1 and ``1/e(x)`` is undefined; it is nested inside the label, so the treated arm has
no control counterfactual; and it fires on 16 rows out of 6.36 million. ``app.data.adapters``
already drops it before it can reach a model, calling it a leaked downstream decision.

Every transaction in this project's data was allowed. There is no logged policy to reweight,
and **an average treatment effect recovered from this data would be a fabrication**.

**What is estimable, and why that is still worth building.** Under a stated cost model both
potential outcomes are deterministic functions of the label and the amount::

    cost(allow) = Y * (A + chargeback_fee)
    cost(block) = (1 - Y) * review_cost

so the true cost of *any* deterministic policy on a labelled split can be computed exactly,
not estimated. That is what :func:`true_policy_cost` does, and it is what makes this module
useful: with ground truth in hand, an off-policy estimator can be **validated** rather than
merely applied. This module therefore simulates a logging policy with a known propensity, runs
the direct method, IPW and the doubly robust estimator against it, and reports each one's bias
against the exact answer.

That is an honest use of the machinery and a genuinely useful one -- a deployment with real
logged decisions needs exactly this estimator, and here it can be checked. What it is not is a
causal effect recovered from observational data. Every function below says which it is.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from app.ml.cost import CostModel

#: Propensities are clipped into ``[EPSILON, 1 - EPSILON]`` before any division. A logging
#: policy that assigns some region of feature space probability zero cannot be reweighted
#: there at any sample size -- that is the positivity assumption, and it fails hard rather
#: than gradually. Clipping bounds the damage; :func:`assert_positivity` refuses the run.
EPSILON = 1e-3

#: How far a propensity may approach 0 or 1 before the run is refused outright. Deliberately
#: looser than EPSILON: clipping a handful of extreme weights is routine, but a policy whose
#: propensities are *deterministic* is not an estimation problem, it is an unanswerable
#: question. This is the guard that ``isFlaggedFraud`` would trip.
POSITIVITY_FLOOR = 1e-6


def assert_positivity(propensity: npt.NDArray[np.float64]) -> None:
    """Raise if any propensity is close enough to 0 or 1 to make reweighting meaningless.

    Called before every estimator that divides by a propensity, rather than trusted. A
    deterministic logging policy produces infinite weights, and the resulting estimate is not a
    wide interval -- it is arbitrary, and it looks like a number.

    Raises:
        ValueError: If any propensity falls outside
            ``[POSITIVITY_FLOOR, 1 - POSITIVITY_FLOOR]``.
    """
    if propensity.size == 0:
        raise ValueError("propensity vector is empty; nothing to evaluate")
    low = float(np.min(propensity))
    high = float(np.max(propensity))
    if low < POSITIVITY_FLOOR or high > 1.0 - POSITIVITY_FLOOR:
        raise ValueError(
            f"positivity violated: propensities span [{low:.3g}, {high:.3g}], outside "
            f"[{POSITIVITY_FLOOR:g}, {1 - POSITIVITY_FLOOR:g}]. A logging policy that acts "
            "deterministically on some region of feature space cannot be reweighted there at "
            "any sample size. This is the failure mode that makes the PaySim isFlaggedFraud "
            "column unusable as a treatment."
        )


@dataclass(frozen=True)
class LoggingPolicy:
    """A simulated historical policy, with the propensity it acted under.

    **Simulated, never recovered.** The propensity is known here because this module wrote it,
    which is precisely what makes the validation in :class:`OpeReport` possible and precisely
    why no causal claim about the real corpus may rest on it.

    Attributes:
        blocked: True where the simulated policy blocked the transaction.
        propensity: ``P(block | x)`` the policy acted under, in ``(0, 1)``.
        description: Plain-language statement of the simulation, printed with every report.
    """

    blocked: npt.NDArray[np.bool_]
    propensity: npt.NDArray[np.float64]
    description: str

    def __post_init__(self) -> None:
        """Validate shape agreement and positivity at construction."""
        if self.blocked.shape != self.propensity.shape:
            raise ValueError(
                f"blocked {self.blocked.shape} and propensity {self.propensity.shape} disagree"
            )
        assert_positivity(self.propensity)


def simulate_logging_policy(
    scores: npt.NDArray[np.float64],
    *,
    aggressiveness: float = 4.0,
    floor: float = 0.05,
    ceiling: float = 0.60,
    seed: int = 42,
) -> LoggingPolicy:
    """Simulate a stochastic historical reviewer whose propensity is known by construction.

    Modelled as a soft version of what a rules-based team plausibly did before any model
    existed: block more often as the risk score rises, but never deterministically, and never
    outside a band that keeps every region of feature space reachable. The band is what makes
    the resulting problem estimable at all -- see :func:`assert_positivity`.

    Args:
        scores: Risk scores in ``[0, 1]``. Higher means the policy blocks more often.
        aggressiveness: How sharply the block probability rises with the score.
        floor: Minimum block probability. Guarantees positivity on the treated arm.
        ceiling: Maximum block probability. Guarantees positivity on the control arm.
        seed: Seed for the simulated draws, logged with every result.
    """
    if not 0.0 < floor < ceiling < 1.0:
        raise ValueError(f"need 0 < floor < ceiling < 1, got floor={floor}, ceiling={ceiling}")
    centred = aggressiveness * (scores - float(np.mean(scores)))
    shaped = 1.0 / (1.0 + np.exp(-centred))
    propensity = floor + (ceiling - floor) * shaped
    rng = np.random.default_rng(seed)
    return LoggingPolicy(
        blocked=rng.random(scores.shape) < propensity,
        propensity=propensity.astype(np.float64),
        description=(
            f"Simulated stochastic reviewer: P(block) = {floor:.2f} + "
            f"{ceiling - floor:.2f} * sigmoid({aggressiveness:.1f} * (score - mean score)), "
            f"seed {seed}. SIMULATED, not recovered from data -- neither corpus records a "
            "historical action. Its only purpose is to give the estimators below a policy "
            "whose propensity is known, so their answers can be checked against exact truth."
        ),
    )


def realised_cost(
    labels: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    blocked: npt.NDArray[np.bool_],
    model: CostModel,
) -> npt.NDArray[np.float64]:
    """Return the per-row cost actually incurred under a given set of decisions.

    The cost model, stated as arithmetic:

    * blocked and legitimate -- a false positive, costing one review.
    * blocked and fraudulent -- a true positive, costing nothing: the loss was averted.
    * allowed and fraudulent -- a false negative: the amount is lost, plus the chargeback fee.
    * allowed and legitimate -- a true negative, costing nothing.

    This is deliberately the *same* arithmetic as :func:`app.ml.cost.cost_at_threshold`, which
    every result from Phase 2 onwards is quoted on. Charging the review to true positives as
    well would be more realistic -- an analyst is paid whatever the queue turns up -- but it
    would silently move the basis, and this phase compares against those published numbers.
    """
    blocked_cost = np.where(labels, 0.0, model.review_cost)
    allowed_cost = np.where(labels, amounts + model.chargeback_fee, 0.0)
    return np.where(blocked, blocked_cost, allowed_cost).astype(np.float64)


def true_policy_cost(
    labels: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    decisions: npt.NDArray[np.bool_],
    model: CostModel,
) -> float:
    """Return the exact mean cost of a deterministic policy. Not an estimate.

    Computable because both potential outcomes are deterministic given ``(label, amount)``
    under a stated cost model. This is the ground truth the estimators below are scored
    against, and its existence is the only reason this module can validate rather than assert.
    """
    return float(np.mean(realised_cost(labels, amounts, decisions, model)))


def direct_method(
    predicted_cost_if_blocked: npt.NDArray[np.float64],
    predicted_cost_if_allowed: npt.NDArray[np.float64],
    decisions: npt.NDArray[np.bool_],
) -> float:
    """Return the modelled cost of a target policy, ignoring the logged outcomes entirely.

    Low variance, and biased by exactly however wrong the outcome model is. Reported beside IPW
    so the two failure modes are visible as a pair rather than one being chosen silently.
    """
    return float(np.mean(np.where(decisions, predicted_cost_if_blocked, predicted_cost_if_allowed)))


def ipw(
    observed_cost: npt.NDArray[np.float64],
    logged: LoggingPolicy,
    decisions: npt.NDArray[np.bool_],
) -> float:
    """Return the inverse-probability-weighted cost of a target policy.

    Unbiased when the propensity is correct -- which here it is by construction, and in a real
    deployment it would not be. High variance: rows the logging policy rarely visited carry
    large weights, and a handful of them can dominate the estimate.
    """
    assert_positivity(logged.propensity)
    clipped = np.clip(logged.propensity, EPSILON, 1.0 - EPSILON)
    action_propensity = np.where(logged.blocked, clipped, 1.0 - clipped)
    agrees = logged.blocked == decisions
    return float(np.mean(np.where(agrees, observed_cost / action_propensity, 0.0)))


def doubly_robust(
    observed_cost: npt.NDArray[np.float64],
    predicted_cost_if_blocked: npt.NDArray[np.float64],
    predicted_cost_if_allowed: npt.NDArray[np.float64],
    logged: LoggingPolicy,
    decisions: npt.NDArray[np.bool_],
) -> float:
    """Return the doubly robust (AIPW) cost of a target policy.

    The direct method prediction, plus an importance-weighted correction for how wrong that
    prediction turned out to be on the rows where the logged policy happened to agree with the
    target policy. Consistent if *either* the outcome model or the propensity model is right,
    which is the property the name refers to -- and the reason it is the estimator a real
    deployment should use, where neither model is trustworthy on its own.
    """
    assert_positivity(logged.propensity)
    clipped = np.clip(logged.propensity, EPSILON, 1.0 - EPSILON)
    action_propensity = np.where(logged.blocked, clipped, 1.0 - clipped)
    predicted_under_target = np.where(
        decisions, predicted_cost_if_blocked, predicted_cost_if_allowed
    )
    predicted_under_logged = np.where(
        logged.blocked, predicted_cost_if_blocked, predicted_cost_if_allowed
    )
    agrees = logged.blocked == decisions
    correction = np.where(agrees, (observed_cost - predicted_under_logged) / action_propensity, 0.0)
    return float(np.mean(predicted_under_target + correction))


@dataclass(frozen=True)
class OpeReport:
    """Three estimators scored against the exact answer.

    The honesty exhibit for Phase 6. Each estimator is asked what a target policy would cost,
    given only the logged decisions and their propensities; the truth is then computed directly
    and the biases reported. An estimator that cannot recover a known answer under a propensity
    it was *handed* has no business being trusted on one it had to estimate.
    """

    policy_name: str
    truth: float
    direct: float
    ipw: float
    doubly_robust: float
    logging_description: str
    rows: int

    def _bias(self, estimate: float) -> float:
        """Return relative bias against the exact cost."""
        return (estimate - self.truth) / self.truth if self.truth else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the registry shape."""
        return {
            "policy": self.policy_name,
            "rows": self.rows,
            "true_cost_per_unit": round(self.truth, 6),
            "direct_method": round(self.direct, 6),
            "ipw": round(self.ipw, 6),
            "doubly_robust": round(self.doubly_robust, 6),
            "relative_bias": {
                "direct_method": round(self._bias(self.direct), 6),
                "ipw": round(self._bias(self.ipw), 6),
                "doubly_robust": round(self._bias(self.doubly_robust), 6),
            },
            "logging_policy": self.logging_description,
            "caveat": (
                "The logging policy is SIMULATED and its propensity is known by construction. "
                "These figures validate the estimators; they are not a causal effect measured "
                "on this corpus, which records no historical action at all."
            ),
        }

    def render(self) -> str:
        """Return the report block."""
        return "\n".join(
            [
                f"Off-policy evaluation of: {self.policy_name}  (n={self.rows:,})",
                f"  true cost per unit   {self.truth:>12,.4f}   (exact, computed from labels)",
                f"  direct method        {self.direct:>12,.4f}   "
                f"bias {100 * self._bias(self.direct):>+7.2f}%",
                f"  IPW                  {self.ipw:>12,.4f}   "
                f"bias {100 * self._bias(self.ipw):>+7.2f}%",
                f"  doubly robust        {self.doubly_robust:>12,.4f}   "
                f"bias {100 * self._bias(self.doubly_robust):>+7.2f}%",
                f"  logging policy: {self.logging_description}",
            ]
        )


def evaluate_policy(
    policy_name: str,
    labels: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    decisions: npt.NDArray[np.bool_],
    predicted_cost_if_blocked: npt.NDArray[np.float64],
    predicted_cost_if_allowed: npt.NDArray[np.float64],
    logged: LoggingPolicy,
    model: CostModel,
) -> OpeReport:
    """Score all three estimators on one target policy against the exact truth."""
    observed = realised_cost(labels, amounts, logged.blocked, model)
    return OpeReport(
        policy_name=policy_name,
        truth=true_policy_cost(labels, amounts, decisions, model),
        direct=direct_method(predicted_cost_if_blocked, predicted_cost_if_allowed, decisions),
        ipw=ipw(observed, logged, decisions),
        doubly_robust=doubly_robust(
            observed,
            predicted_cost_if_blocked,
            predicted_cost_if_allowed,
            logged,
            decisions,
        ),
        logging_description=logged.description,
        rows=int(labels.size),
    )
