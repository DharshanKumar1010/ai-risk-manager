"""Causal cost layer: estimates the financial cost of each block/allow decision.

Phase 6. The track's named bar is an honest false-positive cost per decision, and this is the
layer that produces it. Everything here is an estimate resting on stated assumptions, and every
figure it emits carries them.

Why the DR-Learner collapses on this data, proved rather than asserted
======================================================================

The brief frames this as a treatment-effect problem and asks for inverse probability weighting
on the historical actions. **Neither corpus records an action** -- see :mod:`app.ml.ope` for the
column-by-column evidence. Every transaction here was allowed.

That is not merely an obstacle; it determines what this layer can be. Write the cost model as
arithmetic, matching :func:`app.ml.cost.cost_at_threshold` exactly, with ``Y`` the fraud label,
``A`` the amount, ``r`` the review cost and ``f`` the chargeback fee::

    cost(block | Y) = (1 - Y) * r          a review is paid only when the row was legitimate
    cost(allow | Y) = Y * (A + f)          a missed fraud loses the amount, plus the fee

Take expectations given features ``x``, writing ``p(x) = P(Y = 1 | x)``::

    E[cost(block) | x] = (1 - p(x)) * r
    E[cost(allow) | x] = p(x) * (A + f)

The conditional average treatment effect of blocking -- the object a DR-Learner exists to
estimate -- is therefore::

    tau(x) = E[cost(block) - cost(allow) | x]
           = (1 - p(x)) * r  -  p(x) * (A + f)

**Every term on the right is either known at decision time or is ``p(x)``.** The amount, the
review cost and the fee are all known before the decision; the label is the only random
quantity, and it is binary, so its conditional expectation is exactly ``p(x)``. There is no
residual confounding for a doubly robust correction to remove, because there is no treatment
whose assignment could be confounded. The DR-Learner does not fail here -- it *collapses*,
provably and exactly, onto a cost-weighted plug-in rule driven by a calibrated probability.

Stating that is worth more than hiding it behind machinery that would produce the same numbers.
Two things follow, and both are built:

**The decision rule stops being a global threshold.** Blocking pays when ``tau(x) < 0``, that is
when ``p(x) > r / (A + f + r)``. The cut-off *depends on the amount*: a large transaction earns
a block at a far lower probability than a small one. A single global threshold on ``p`` is the
special case where every amount is identical, and this corpus is nothing like that -- Phase 2
measured Tier-1 catching 24.6% of fraud by count and only 14.6% by value.

**There is still something to learn.** The plug-in above reaches the expected loss through
``p(x) * (A + f)``, a classifier fitted to minimise log-loss over *counts*. Regressing the
realised loss ``Y * (A + f)`` on the same features directly optimises squared error over
*value*, which weights a large fraud more heavily than a small one during fitting rather than
only afterwards. Those are different models and they can rank differently. :class:`LossModel`
is that regression, and the run measures whether it beats the plug-in. That is the
"cost-sensitive training, not just cost-sensitive thresholding" opening BUILD_LOG recorded.

What this layer does NOT do
===========================

It does not recover a causal effect from observational data, it does not estimate what a real
historical reviewer would have done, and it does not know the true cost of a false positive.
It converts a calibrated probability and a stated cost model into a decision, and reports how
sensitive that decision is to the stated part.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.core.audit import Decision
from app.ml.cost import CostModel
from app.ml.registry import artifact_path
from app.models.meta_features import TimeBlocks, assert_forward_chaining, time_blocks

logger = logging.getLogger("riskiq.cost")

#: Forward-chaining blocks for the out-of-fold loss regression. Matches the Phase 5 out-of-fold
#: scheme so the two phases carry the same handicap rather than two different ones.
LOSS_OOF_BLOCKS = 5

#: LightGBM parameters for the loss regression. Deliberately close to Tier-1's, so a difference
#: between the two models is attributable to the *target* -- realised loss against the binary
#: label -- rather than to a hyperparameter search this phase did not run.
#:
#: ``objective`` is the one deliberate departure. L2 on a target that is zero for 96.5% of rows
#: and lognormal-tailed on the rest is dominated by a handful of large frauds; Tweedie is built
#: for exactly this zero-inflated, positive-skewed shape and is what a claims-cost model uses.
LOSS_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "tweedie",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

#: One reproducibility caveat travels with these parameters. ``bagging_fraction`` and
#: ``feature_fraction`` both draw against row position, so the fitted booster depends on the
#: order its training rows arrive in, not only on which rows they are. ``deterministic`` and a
#: logged seed make a run repeatable given the same parquet; they do not make it invariant to a
#: permutation of that parquet. Disabling both sampling parameters removes the dependence
#: entirely and was measured to do so, at the usual cost in variance.

#: Ceiling on an amount this layer will price, mirroring ``transactions.amount`` --
#: ``NUMERIC(20, 4)``, so sixteen digits before the point. Paired with a floor of zero: the
#: database has ``ck_transactions_amount_non_negative``, but a scoring request need not be
#: persisted before it reaches the cost arithmetic.
MAX_SCOREABLE_AMOUNT = 1e16

#: Boosting rounds for the loss regression, and the early-stopping patience.
LOSS_NUM_ROUNDS = 600
LOSS_EARLY_STOPPING = 50


@dataclass(frozen=True)
class DecisionCost:
    """What one decision on one transaction is expected to cost.

    The per-decision object the track's brief names. Note the deliberate rename: the brief calls
    this ``CostEstimate``, but :class:`app.ml.cost.CostEstimate` already exists and means
    something else entirely -- the cost of a *population* of decisions at one threshold, imported
    by all four training drivers. Two different shapes under one name in one project is a bug
    waiting to be written, so the per-decision type is ``DecisionCost``.

    Attributes:
        expected_cost: Cost of the decision actually taken.
        cost_if_blocked: Expected cost had this transaction been blocked.
        cost_if_allowed: Expected cost had it been allowed through.
        fraud_probability: The calibrated ``p(x)`` both arms are computed from.
        amount: The transaction amount, which sets how much a miss costs.
        decision: The decision these figures describe.
        assumptions: Plain-language statement of what the numbers rest on.
    """

    expected_cost: float
    cost_if_blocked: float
    cost_if_allowed: float
    fraud_probability: float
    amount: float
    decision: Decision
    assumptions: list[str] = field(default_factory=list)

    @property
    def expected_saving_from_blocking(self) -> float:
        """Return how much blocking is expected to save. Negative means allowing is cheaper."""
        return self.cost_if_allowed - self.cost_if_blocked

    def to_audit_dict(self) -> dict[str, Any]:
        """Return the JSON shape **for the server-side audit trail only. Never a response body.**

        Every field here is an evasion oracle, and together they are complete.
        ``fraud_probability`` is the model output; ``cost_if_blocked`` is ``(1-p)*r``, so a
        caller holding both recovers the review cost; ``cost_if_allowed`` is ``p*(A+f)``, which
        with a caller-chosen amount recovers the chargeback fee; and the **sign** of
        ``expected_saving_from_blocking`` is literally the decision boundary, so a handful of
        probe transactions binary-search the largest amount that evades review at a given risk
        score. ``assumptions`` prints the cost matrix in plain English.

        ``app.core.audit`` already carries this warning on ``top_features``; this object is
        strictly worse, because ``top_features`` says which features mattered while this says
        how far from the boundary the caller is and in which direction to move.

        Phase 7 must not echo any field of this to a transacting party. A scoring response may
        carry the decision and an opaque audit id; anything numeric from here stays server-side.
        """
        return {
            "decision": self.decision,
            "expected_cost": round(self.expected_cost, 4),
            "cost_if_blocked": round(self.cost_if_blocked, 4),
            "cost_if_allowed": round(self.cost_if_allowed, 4),
            "expected_saving_from_blocking": round(self.expected_saving_from_blocking, 4),
            "fraud_probability": round(self.fraud_probability, 6),
            "amount": round(self.amount, 2),
            "assumptions": list(self.assumptions),
        }


def expected_cost_if_blocked(
    probability: npt.NDArray[np.float64], model: CostModel
) -> npt.NDArray[np.float64]:
    """Return ``(1 - p) * review_cost`` -- a review is paid only on the legitimate rows."""
    return (1.0 - probability) * model.review_cost


def expected_cost_if_allowed(
    probability: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
) -> npt.NDArray[np.float64]:
    """Return ``p * (amount + chargeback_fee)`` -- the plug-in expected loss."""
    return probability * (amounts + model.chargeback_fee)


def amount_aware_threshold(
    amounts: npt.NDArray[np.float64], model: CostModel
) -> npt.NDArray[np.float64]:
    """Return the per-transaction probability at which blocking starts to pay.

    ``r / (A + f + r)``, the break-even from the module docstring. Reported alongside the
    policies because it is the clearest statement of what cost-aware ranking actually changes:
    not how transactions are scored, but how high the bar is set for each one.
    """
    return model.review_cost / (amounts + model.chargeback_fee + model.review_cost)


@dataclass(frozen=True)
class LossModel:
    """A cross-fitted regression of realised fraud loss on transaction features.

    Predicts ``E[Y * (amount + fee) | x]`` directly, rather than reaching it by multiplying a
    classifier's ``p(x)`` by the amount. Fitted on the loss target, so large frauds carry more
    weight during fitting; the plug-in only weights them afterwards, at decision time.

    Whether that matters is an empirical question the run answers, not an assumption this class
    makes. It loses to the plug-in on this corpus if the classifier's ranking is simply better.

    Attributes:
        booster: The fitted LightGBM booster.
        feature_names: Columns it consumes, in order.
        best_iteration: Early-stopped round, or 0 when the full schedule ran.
        chargeback_fee: The fee folded into the target it was fitted on. Recorded because a
            model fitted against one fee cannot be read against another.
    """

    booster: Any
    feature_names: tuple[str, ...]
    best_iteration: int
    chargeback_fee: float

    def predict(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return predicted expected loss, floored at zero.

        Tweedie predictions are non-negative by construction, but the floor is kept because a
        negative expected loss would silently invert a ranking rather than fail loudly.
        """
        iteration = self.best_iteration or 0
        raw = self.booster.predict(matrix, num_iteration=iteration or None)
        return np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)

    def to_dict(self) -> dict[str, Any]:
        """Return the sidecar shape."""
        return {
            "feature_names": list(self.feature_names),
            "best_iteration": self.best_iteration,
            "chargeback_fee": self.chargeback_fee,
            "hyperparameters": dict(LOSS_PARAMS),
            "num_rounds": LOSS_NUM_ROUNDS,
            "early_stopping": LOSS_EARLY_STOPPING,
        }


def loss_target(
    labels: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
) -> npt.NDArray[np.float64]:
    """Return the realised loss each row would incur if allowed through.

    Zero for legitimate rows, ``amount + fee`` for fraudulent ones. This is an *observed*
    quantity on every row of this data, precisely because every transaction was allowed --
    which is the one thing the missing treatment variable makes easier rather than harder.
    """
    return np.where(labels, amounts + model.chargeback_fee, 0.0).astype(np.float64)


def fit_loss_model(
    matrix: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    feature_names: Sequence[str],
    seed: int,
    chargeback_fee: float,
    validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    rounds: int = LOSS_NUM_ROUNDS,
) -> LossModel:
    """Fit the loss regression, early-stopping on ``validation`` when one is supplied.

    ``chargeback_fee`` is recorded on the returned model rather than inferred, because a
    model fitted against one fee produces meaningless predictions when read against another.
    """
    import lightgbm as lgb

    params = dict(LOSS_PARAMS, seed=seed)
    train_set = lgb.Dataset(matrix, label=target, feature_name=list(feature_names))
    callbacks: list[Callable[..., Any]] = []
    valid_sets = []
    if validation is not None:
        valid_matrix, valid_target = validation
        valid_sets = [
            lgb.Dataset(
                valid_matrix,
                label=valid_target,
                feature_name=list(feature_names),
                reference=train_set,
            )
        ]
        callbacks = [lgb.early_stopping(LOSS_EARLY_STOPPING, verbose=False)]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return LossModel(
        booster=booster,
        feature_names=tuple(feature_names),
        best_iteration=int(getattr(booster, "best_iteration", 0) or 0),
        chargeback_fee=chargeback_fee,
    )


def out_of_fold_loss(
    frame: pd.DataFrame,
    matrix: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    feature_names: Sequence[str],
    seed: int,
    chargeback_fee: float,
    blocks: int = LOSS_OOF_BLOCKS,
) -> npt.NDArray[np.float64]:
    """Return out-of-fold loss predictions over a training split.

    Forward-chaining: each block is scored by a model fitted only on strictly earlier blocks.
    In-sample predictions on the fitting split would make the loss model look better than the
    plug-in for no reason other than having memorised the rows, and the comparison between them
    is the entire question this phase asks.

    The first block has nothing earlier to fit on and is left unscored, returned as NaN. Callers
    must drop those rows rather than treat them as zero-loss predictions.
    """
    assignment: TimeBlocks = time_blocks(frame["event_time"], block_count=blocks)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    for block in range(2, assignment.block_count + 1):
        assert_forward_chaining(assignment, frame["event_time"], block)
        fit_mask = assignment.rows_before(block)
        score_mask = assignment.rows_in(block)
        model = fit_loss_model(
            matrix[fit_mask],
            target[fit_mask],
            feature_names=feature_names,
            seed=seed,
            chargeback_fee=chargeback_fee,
            rounds=LOSS_NUM_ROUNDS // 2,
        )
        predictions[score_mask] = model.predict(matrix[score_mask])
        logger.info(
            "loss OOF block %d/%d: fitted on %d rows, scored %d",
            block,
            assignment.block_count,
            int(fit_mask.sum()),
            int(score_mask.sum()),
        )
    return predictions


@dataclass(frozen=True)
class CostPolicy:
    """Ranks transactions by expected cost saving, and decides against a capacity cap.

    Three ranking strategies share this class, so they differ only in the score they rank by and
    can therefore be compared like for like:

    * ``probability`` -- rank by ``p(x)``. The Phase 2 baseline; cost enters only through where
      the threshold lands.
    * ``plug_in`` -- rank by ``p(x) * (A + f) - (1 - p(x)) * r``, the closed form derived in the
      module docstring.
    * ``learned_loss`` -- rank by ``mu(x) - (1 - p(x)) * r``, substituting the loss regression
      for the plug-in's first term.

    Attributes:
        strategy: Which of the three above.
        cost_model: The stated costs. Every figure this policy produces depends on it.
        threshold: Operating point on the ranking score, chosen on validation.
        threshold_criterion: How that point was chosen, carried into the report verbatim.
        loss_model: The regression, present only for ``learned_loss``.
    """

    strategy: str
    cost_model: CostModel
    threshold: float
    threshold_criterion: str
    loss_model: LossModel | None = None

    def ranking_score(
        self,
        probability: npt.NDArray[np.float64],
        amounts: npt.NDArray[np.float64],
        predicted_loss: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the score this policy ranks by. Higher means block sooner.

        Raises:
            ValueError: If the strategy is unknown, or if ``learned_loss`` is asked to rank
                without a prediction vector.
        """
        if self.strategy == "probability":
            return probability
        blocked = expected_cost_if_blocked(probability, self.cost_model)
        if self.strategy == "plug_in":
            return expected_cost_if_allowed(probability, amounts, self.cost_model) - blocked
        if self.strategy == "learned_loss":
            if predicted_loss is None:
                raise ValueError("learned_loss ranking needs predicted_loss")
            return predicted_loss - blocked
        raise ValueError(f"unknown strategy {self.strategy!r}")

    def estimate_cost(
        self,
        amount: float,
        fraud_probability: float,
        decision: Decision,
        predicted_loss: float | None = None,
    ) -> DecisionCost:
        """Return the expected cost of one decision on one transaction.

        The brief specifies ``estimate_cost(transaction, decision)``. The probability is passed
        in rather than computed here on purpose: scoring belongs to Tier-1, and a cost layer that
        carried its own second scorer would let the two drift apart silently, so that the number
        the audit trail records would not be the number the decision was made on.

        Args:
            amount: The transaction amount.
            fraud_probability: Calibrated ``p(x)`` from the scoring layer.
            decision: The decision taken. ``review`` is priced as ``block``, since both put the
                transaction in front of a human and neither lets it complete.
            predicted_loss: The loss regression's estimate, for ``learned_loss`` policies.
        """
        if not 0.0 <= fraud_probability <= 1.0:
            raise ValueError(f"fraud_probability must be in [0, 1], got {fraud_probability}")
        # The amount needs the same guard and did not have it. A negative amount makes
        # cost_if_allowed negative, drives the expected saving far below any threshold, and so
        # allows the transaction however fraudulent the model believes it to be -- one request
        # turns the cost layer off. The ceiling mirrors the transactions.amount column, which is
        # NUMERIC(20, 4); a scoring call does not have to be persisted first to reach here.
        if not 0.0 <= amount <= MAX_SCOREABLE_AMOUNT:
            raise ValueError(
                f"amount must be in [0, {MAX_SCOREABLE_AMOUNT:g}], got {amount}. A negative "
                "amount would invert the ranking score and suppress blocking."
            )
        if self.strategy == "learned_loss" and predicted_loss is None:
            # ranking_score refuses this; so must the audit path, or the number recorded against
            # a decision is not the number the decision was made on.
            raise ValueError(
                "a learned_loss policy needs predicted_loss to estimate a cost; falling back to "
                "the plug-in here would record a different figure than the one that decided"
            )
        probability = np.asarray([fraud_probability], dtype=np.float64)
        amounts = np.asarray([amount], dtype=np.float64)
        blocked = float(expected_cost_if_blocked(probability, self.cost_model)[0])
        if predicted_loss is not None:
            allowed = float(predicted_loss)
        else:
            allowed = float(expected_cost_if_allowed(probability, amounts, self.cost_model)[0])
        return DecisionCost(
            expected_cost=blocked if decision in ("block", "review") else allowed,
            cost_if_blocked=blocked,
            cost_if_allowed=allowed,
            fraud_probability=fraud_probability,
            amount=amount,
            decision=decision,
            assumptions=self.cost_model.assumptions(),
        )

    def decide(
        self,
        probability: npt.NDArray[np.float64],
        amounts: npt.NDArray[np.float64],
        predicted_loss: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return which transactions this policy flags at its operating point."""
        score = self.ranking_score(probability, amounts, predicted_loss)
        return np.asarray(score >= self.threshold, dtype=np.bool_)

    def save(self, model_id: str, directory: Path) -> Path:
        """Write the policy sidecar, and the loss booster when there is one.

        Paths go through :func:`app.ml.registry.artifact_path` rather than being joined here.
        That guard exists because ``Path("models/artifacts") / "C:evil.json"`` discards the base
        on Windows, and the same hole reached two tiers before it was closed.
        """
        directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_id": model_id,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "threshold_criterion": self.threshold_criterion,
            "cost_model": {
                "review_cost": self.cost_model.review_cost,
                "chargeback_fee": self.cost_model.chargeback_fee,
                "units": self.cost_model.units,
                "unit_noun": self.cost_model.unit_noun,
            },
            "assumptions": self.cost_model.assumptions(),
            "loss_model": self.loss_model.to_dict() if self.loss_model else None,
        }
        if self.loss_model is not None:
            booster_path = artifact_path(model_id, directory, ".txt")
            self.loss_model.booster.save_model(str(booster_path))
        sidecar = artifact_path(model_id, directory, ".meta.json")
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return sidecar

    @classmethod
    def load(cls, model_id: str, directory: Path) -> "CostPolicy":
        """Rebuild a policy from its sidecar, and its booster when the strategy needs one."""
        import lightgbm as lgb

        payload = json.loads(
            artifact_path(model_id, directory, ".meta.json").read_text(encoding="utf-8")
        )
        costs = payload["cost_model"]
        loss_model = None
        if payload.get("loss_model"):
            spec = payload["loss_model"]
            booster = lgb.Booster(model_file=str(artifact_path(model_id, directory, ".txt")))
            loss_model = LossModel(
                booster=booster,
                feature_names=tuple(spec["feature_names"]),
                best_iteration=int(spec["best_iteration"]),
                chargeback_fee=float(spec["chargeback_fee"]),
            )
        return cls(
            strategy=payload["strategy"],
            cost_model=CostModel(
                review_cost=float(costs["review_cost"]),
                chargeback_fee=float(costs["chargeback_fee"]),
                units=costs["units"],
                unit_noun=costs["unit_noun"],
            ),
            threshold=float(payload["threshold"]),
            threshold_criterion=payload["threshold_criterion"],
            loss_model=loss_model,
        )
