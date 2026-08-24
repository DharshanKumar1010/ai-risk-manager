"""Meta-learner: fuses the Tier-1/2/3 signals into one calibrated risk score.

Phase 5 implements this layer: an XGBoost classifier over the three tier signals plus the
original engineered features, probability-calibrated, with TreeSHAP supplying per-prediction
attribution, exposing

    predict(transaction, tier1, tier2, tier3) -> MetaResult

Raw XGBoost margins are not probabilities. Calibration is required, not optional -- an
uncalibrated score undermines the honest-metrics claim this project rests on.

**Where the leakage discipline lives.** Not here. ``meta_features`` owns the out-of-fold Tier-1
scoring, the Tier-3 snapshot rebuild and the deny list; this module owns the model, the
calibrator and the serving contract. The split is deliberate: the honesty of the numbers is a
property of how the matrix was built, and keeping that in one auditable place is worth the
extra module.

**Two implementation choices worth stating, because both look like shortcuts and are not.**

*TreeSHAP comes from XGBoost, not from the ``shap`` package.* ``booster.predict(...,
pred_contribs=True)`` runs the same TreeSHAP algorithm and was verified bit-identical to
``shap.TreeExplainer.shap_values`` (maximum absolute difference 0.0) on this project's model
shape. It also avoids a real problem: importing ``shap`` emits ``PendingDeprecationWarning``
from matplotlib at import time, and this repository runs pytest under
``filterwarnings = ["error"]``, so any test importing it fails at collection. Tier-1 already
takes the same route through LightGBM's native ``pred_contrib``.

*Calibration is Platt scaling, stored as two floats.* Isotonic regression is piecewise-constant
and therefore ties scores, which moves PR-AUC by an amount that is an artefact of step width
rather than a property of the model -- unacceptable when PR-AUC is the headline. A sigmoid is
strictly monotone and leaves the ranking, and so the PR-AUC, exactly intact. It also serialises
as ``(a, b)`` in plain JSON, honouring the no-pickle rule Tier-2 and Tier-3 both state: loading
a pickle executes arbitrary code, and this path becomes reachable from a Phase 7 endpoint.

**Security.** ``top_features`` is an evasion oracle. Feature attribution tells the recipient
which signals drove a decision, which is precisely what an adversary needs to avoid triggering
them next time. It must be authenticated, restricted to internal reviewers, and never returned
to the transacting party -- the same constraint BUILD_LOG records for Tier-1's ``explain``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, ClassifierMixin

from app.core.audit import Decision
from app.data.raw_spec import SourceDataset
from app.data.schema import TransactionFeatures
from app.ml.registry import artifact_path
from app.models.meta_features import (
    ABSTENTION_SENTINEL,
    RANDOM_SEED,
    block_feature_names,
    require_clean_feature_names,
)
from app.models.tier1_anomaly import Tier1Result
from app.models.tier2_behavioral import Tier2Result
from app.models.tier3_graph import Tier3Result

logger = logging.getLogger("riskiq.meta")

#: Serving budget. Generous relative to Tier-1's 6.4ms p95 because the meta-learner runs after
#: all three tiers and its own tree ensemble is shallow; the budget exists to catch a regression,
#: not to describe expected latency.
LATENCY_BUDGET_P95_MS = 50.0

#: Deliberately shallow. The strongest input is already the output of a 1,917-round LightGBM,
#: and a deep meta-learner spends its capacity refitting that model's residual noise rather than
#: learning how to combine layers. ``n_jobs`` is pinned for the same reason LightGBM's
#: ``num_threads`` is: models/registry.json claims reproducibility for every model in it.
XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 20.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "nthread": 4,
    "seed": RANDOM_SEED,
    "verbosity": 0,
}

XGBOOST_NUM_ROUNDS = 2_000
XGBOOST_EARLY_STOPPING = 100

#: Rounds for every ablation variant. Fixed and equal across variants so the comparison isolates
#: the feature blocks rather than how much fitting each variant happened to get.
ABLATION_NUM_ROUNDS = 300
ABLATION_MAX_DEPTH = 3


class MetaResult(BaseModel):
    """One fused decision, with its explanation and its provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated P(fraud). Not the raw XGBoost margin."
    )
    decision: Decision
    top_features: tuple[tuple[str, float], ...] = Field(
        default=(),
        description="Top contributing features by absolute TreeSHAP value, in margin space. "
        "An evasion oracle: internal reviewers only, never the transacting party. A tuple "
        "rather than a list because pydantic's frozen=True is shallow -- it blocks attribute "
        "assignment, not mutation of a list held in a field, and this value ends up on an "
        "audit record whose whole purpose is tamper-evidence.",
    )
    model_version: str
    latency_ms: float = Field(ge=0.0)
    degraded: bool = Field(
        default=False, description="True when a tier abstained or was unavailable."
    )
    degraded_reason: str | None = None

    def public(self) -> dict[str, Any]:
        """Return the fields safe to serialise to the transacting party.

        ``top_features`` is deliberately absent, and this method exists so that omitting it is
        something a caller has to *undo* rather than something they have to remember. Returning
        attribution to whoever submitted the transaction tells them which signals to avoid next
        time, which the security checklist treats as an evasion oracle and the track treats as a
        disqualification rather than a hardening preference.

        The internal reviewer path serialises the full object instead, behind authentication.
        """
        return {
            "probability": self.probability,
            "decision": self.decision,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Platt scaling: ``P = 1 / (1 + exp(a * margin + b))``.

    Two floats, so the artefact is plain JSON and no pickle is ever loaded. Strictly monotone in
    the margin, which is what lets the calibrated probability be reported alongside a PR-AUC
    computed on the raw margin without the two disagreeing about the ranking.
    """

    a: float
    b: float

    @classmethod
    def from_sklearn(cls, fitted: Any) -> "SigmoidCalibrator":
        """Extract the two parameters from a fitted ``CalibratedClassifierCV``.

        Raises:
            ValueError: If the fit produced anything other than exactly one calibrator, which
                would mean an ensemble was built and no single ``(a, b)`` describes it.
        """
        calibrators = list(fitted.calibrated_classifiers_)
        if len(calibrators) != 1:
            raise ValueError(
                f"expected exactly one calibrator, got {len(calibrators)}. Fit with "
                "FrozenEstimator so no cross-validated ensemble is built."
            )
        pair = calibrators[0].calibrators[0]
        return cls(a=float(pair.a_), b=float(pair.b_))

    def apply(self, margin: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map raw margins to calibrated probabilities."""
        return np.asarray(
            1.0 / (1.0 + np.exp(self.a * np.asarray(margin, dtype="float64") + self.b)),
            dtype="float64",
        )

    def to_dict(self) -> dict[str, float]:
        """Return the JSON-serialisable form."""
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SigmoidCalibrator":
        """Rebuild from :meth:`to_dict` output."""
        return cls(a=float(payload["a"]), b=float(payload["b"]))


@dataclass(frozen=True)
class MetaInputSpec:
    """What the meta-learner reads, and everything needed to reproduce it.

    Attributes:
        source_dataset: Always ``ieee_cis`` -- see ``meta_features.require_ieee_cis``.
        blocks: The retained feature blocks, after the ablation.
        feature_names: The matrix columns, in order.
        tier_model_versions: Upstream ``model_id`` per layer, so a past decision traces back to
            the exact three models that produced its inputs.
        upstream_feature_versions: The ``fv_``/``gv_`` hashes behind those inputs. Three separate
            namespaces, deliberately: the pipeline's, Tier-2's and Tier-3's are different
            definitions and collapsing them into one field would be a false claim.
        tier3_prefix: Which Tier-3 variant is in use -- the main graph or the non-circular
            control.
        sentinel: The abstention value, recorded so a reader of the artefact does not have to
            guess what a -1.0 in a tier column means.
    """

    source_dataset: SourceDataset
    blocks: tuple[str, ...]
    feature_names: tuple[str, ...]
    tier_model_versions: Mapping[str, str]
    upstream_feature_versions: Mapping[str, str]
    tier3_prefix: str = "tier3_"
    sentinel: float = ABSTENTION_SENTINEL

    def __post_init__(self) -> None:
        """Refuse a spec whose feature set contains a denied column."""
        require_clean_feature_names(self.feature_names)

    def transform(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return the model matrix for an assembled frame.

        Raises:
            ValueError: If a named feature is absent. Never fills a missing column -- an absent
                feature is a mismatch between this spec and the frame, and guessing a value for
                it would hide that.
        """
        absent = [name for name in self.feature_names if name not in frame.columns]
        if absent:
            raise ValueError(f"assembled frame is missing feature(s): {absent}")
        return frame.loc[:, list(self.feature_names)].to_numpy(dtype="float64")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form for the sidecar."""
        return {
            "source_dataset": self.source_dataset,
            "blocks": list(self.blocks),
            "feature_names": list(self.feature_names),
            "tier_model_versions": dict(self.tier_model_versions),
            "upstream_feature_versions": dict(self.upstream_feature_versions),
            "tier3_prefix": self.tier3_prefix,
            "sentinel": self.sentinel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaInputSpec":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            source_dataset=payload["source_dataset"],
            blocks=tuple(payload["blocks"]),
            feature_names=tuple(payload["feature_names"]),
            tier_model_versions=dict(payload["tier_model_versions"]),
            upstream_feature_versions=dict(payload["upstream_feature_versions"]),
            tier3_prefix=payload.get("tier3_prefix", "tier3_"),
            sentinel=float(payload.get("sentinel", ABSTENTION_SENTINEL)),
        )

    def feature_version(self) -> str:
        """Return this input definition's own hash.

        Its own, not an upstream one. The meta-learner's input genuinely is a new feature
        definition -- three model outputs joined to the engineered vector -- and reusing the
        pipeline's ``fv_`` would claim these rows were produced by a definition that never
        described them. Tier-2 and Tier-3 each mint their own for the same reason.
        """
        from app.data.feature_store import FeatureDefinition, digest_encoders

        return FeatureDefinition(
            source_dataset=self.source_dataset,
            feature_names=self.feature_names,
            parameters={
                "blocks": list(self.blocks),
                "tier_model_versions": dict(self.tier_model_versions),
                "upstream_feature_versions": dict(self.upstream_feature_versions),
                "tier3_prefix": self.tier3_prefix,
                "sentinel": self.sentinel,
            },
            encoder_digest=digest_encoders({}),
        ).feature_version


@dataclass
class MetaModel:
    """The fitted fusion layer: an XGBoost booster plus its calibrator and thresholds."""

    model_id: str
    spec: MetaInputSpec
    booster: Any
    calibrator: SigmoidCalibrator
    review_threshold: float
    block_threshold: float
    algorithm: str = "xgboost + platt-scaled"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    #: Trees actually used at scoring time. Early stopping picks an iteration count on the
    #: validation set, and ``Booster.predict`` then ignores it unless it is passed back in
    #: explicitly -- so a model early-stopped at round 2 was being scored with every round it
    #: happened to train before stopping, i.e. 100 rounds past its own selected optimum.
    #: Scoring the selected model is what early stopping means; leaving this None scores the
    #: whole ensemble and silently discards that selection.
    best_iteration: int | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the matrix columns, in order."""
        return self.spec.feature_names

    def _dmatrix(self, matrix: npt.NDArray[np.float64]) -> Any:
        """Wrap a matrix, carrying the feature names so SHAP output is nameable."""
        import xgboost as xgb

        return xgb.DMatrix(matrix, feature_names=list(self.spec.feature_names))

    @property
    def _iteration_range(self) -> tuple[int, int]:
        """Return the tree range to score with: the validation-selected model, or all of it."""
        if self.best_iteration is None:
            return (0, 0)  # xgboost's sentinel for "every tree"
        return (0, self.best_iteration + 1)

    def margin_frame(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return uncalibrated log-odds for an assembled frame."""
        return self.margins(self.spec.transform(frame))

    def margins(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return uncalibrated log-odds for a prepared matrix."""
        return np.asarray(
            self.booster.predict(
                self._dmatrix(matrix),
                output_margin=True,
                iteration_range=self._iteration_range,
            ),
            dtype="float64",
        )

    def score_frame(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return calibrated probabilities for an assembled frame.

        Used by evaluation and by the training run, not by serving.
        """
        return self.calibrator.apply(self.margin_frame(frame))

    def contributions(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return TreeSHAP contributions, shape ``(n, k + 1)``.

        The trailing column is the base value, so each row sums to that row's raw margin. The
        contributions are in **margin space**; because the calibrator is strictly monotone they
        explain the ordering of the shipped probability too, which is the claim being made when
        they are shown next to one.
        """
        return np.asarray(
            self.booster.predict(
                self._dmatrix(matrix),
                pred_contribs=True,
                iteration_range=self._iteration_range,
            ),
            dtype="float64",
        )

    def explain(self, row: npt.NDArray[np.float64], top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top ``top_k`` contributing features for one row, by absolute value."""
        contributions = self.contributions(np.asarray(row, dtype="float64").reshape(1, -1))[0]
        named = list(zip(self.spec.feature_names, contributions[:-1].tolist(), strict=True))
        named.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return named[:top_k]

    def decide(self, probability: float) -> Decision:
        """Map a calibrated probability to an action.

        Two thresholds, both chosen on validation by cost: below ``review_threshold`` the
        transaction is allowed, at or above ``block_threshold`` it is blocked, and between them
        it goes to a human. The middle band is the point of the cost layer -- a false positive
        that costs an analyst three minutes is a different thing from one that declines a good
        customer.
        """
        if probability >= self.block_threshold:
            return "block"
        if probability >= self.review_threshold:
            return "review"
        return "allow"

    def build_vector(
        self,
        transaction: TransactionFeatures,
        tier1: Tier1Result | None,
        tier2: Tier2Result | None,
        tier3: Tier3Result | None,
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Assemble one serving row, returning the matrix row and any degradation reasons.

        Abstentions become the sentinel *and* set their block's indicator to 0. A zero in the
        score itself would read as "this layer looked and found nothing wrong", which is a
        fabrication about a layer that did not look.

        Raises:
            ValueError: If an engineered feature is missing from ``transaction.features``, or if
                the transaction's ``feature_version`` does not match the one this model was
                fitted against.
        """
        expected = self.spec.upstream_feature_versions.get("pipeline")
        if expected is not None and transaction.feature_version != expected:
            raise ValueError(
                f"feature_version mismatch: transaction carries {transaction.feature_version!r}, "
                f"model was fitted against {expected!r}. Scoring across feature definitions "
                "would silently compare different quantities."
            )

        reasons: list[str] = []
        values: dict[str, float] = {}
        prefix = self.spec.tier3_prefix

        for name in self.spec.feature_names:
            if name == "tier1_score":
                if tier1 is None:
                    values[name] = self.spec.sentinel
                    reasons.append("tier1 unavailable")
                else:
                    values[name] = float(tier1.score)
            elif name == "tier2_error":
                error = None if tier2 is None else tier2.reconstruction_error
                values[name] = self.spec.sentinel if error is None else float(error)
            elif name == "tier2_is_scoreable":
                scoreable = tier2 is not None and tier2.is_scoreable
                values[name] = 1.0 if scoreable else 0.0
                if not scoreable:
                    reasons.append("tier2 abstained" if tier2 is not None else "tier2 unavailable")
            elif name == "tier2_sequence_length":
                values[name] = 0.0 if tier2 is None else float(tier2.sequence_length)
            elif name == f"{prefix}ring_risk_score":
                score = None if tier3 is None else tier3.ring_risk_score
                values[name] = self.spec.sentinel if score is None else float(score)
            elif name == f"{prefix}is_scoreable":
                scoreable = tier3 is not None and tier3.is_scoreable
                values[name] = 1.0 if scoreable else 0.0
                if not scoreable:
                    reasons.append("tier3 abstained" if tier3 is not None else "tier3 unavailable")
            elif name == f"{prefix}seen_not_ringed":
                # The two Tier-3 abstention reasons are different states and are encoded
                # differently: an account the snapshot saw but placed in no qualifying ring is
                # not an account the snapshot never saw.
                from app.models.tier3_graph import ABSTAIN_BELOW_MIN_RING

                seen = (
                    tier3 is not None
                    and not tier3.is_scoreable
                    and tier3.abstention_reason == ABSTAIN_BELOW_MIN_RING
                )
                values[name] = 1.0 if seen else 0.0
            elif name == f"{prefix}is_ring_member":
                values[name] = 1.0 if tier3 is not None and tier3.is_ring_member else 0.0
            elif name == f"{prefix}ring_size":
                values[name] = (
                    float(tier3.ring_size)
                    if tier3 is not None and tier3.is_scoreable
                    else self.spec.sentinel
                )
            elif name.startswith(prefix):
                # Ring topology. Not reachable from a Tier3Result today -- serving it needs
                # Tier-3's score table extended to carry a per-account feature vector. A model
                # retaining this block cannot be served until that lands, which is why the
                # ablation keeps it as its own retirable block.
                values[name] = self.spec.sentinel
                reasons.append(f"{name} unavailable at serving")
            else:
                if name not in transaction.features:
                    raise ValueError(
                        f"engineered feature {name!r} missing from the transaction. A missing "
                        "feature is an error, never a zero: zero is a real value for most of "
                        "these columns and the model cannot tell the two apart."
                    )
                raw = transaction.features[name]
                if isinstance(raw, str):
                    raise ValueError(
                        f"engineered feature {name!r} is {raw!r}, a string. The meta matrix is "
                        "float-only, so this means the feature vector was built against a "
                        "different definition."
                    )
                # A null is a *measurement*, not an absent feature, and the two are handled
                # differently on purpose. `seconds_since_prior_txn` is null for an account's
                # first transaction; `amount_zscore_vs_own_history` is null until there is a
                # history to compare against. The training matrix carried those through as NaN
                # and the booster learned its own default direction for them, so serving must
                # present NaN too. Imputing a number here would fabricate a history the account
                # does not have -- and on IEEE-CIS, where 76% of rows carry no identity record,
                # missingness is one of the more informative things the model sees.
                values[name] = float("nan") if raw is None else float(raw)

        row = np.array([values[name] for name in self.spec.feature_names], dtype="float64")
        return row, reasons

    def predict(
        self,
        transaction: TransactionFeatures,
        tier1: Tier1Result | None = None,
        tier2: Tier2Result | None = None,
        tier3: Tier3Result | None = None,
        *,
        explain: bool = True,
    ) -> MetaResult:
        """Fuse the three tier signals into one calibrated, explained decision.

        Args:
            transaction: The row being scored.
            tier1: Tier-1's result, or None if that layer was unavailable.
            tier2: Tier-2's result, or None. An abstention is not the same as unavailable.
            tier3: Tier-3's result, or None.
            explain: Whether to compute TreeSHAP attribution. Defaults to True because the
                phase brief requires the top three contributors stored with every prediction,
                and the audit trail is where they are stored. Pass False on a path that will
                discard them: attribution is a second full pass over the ensemble, roughly
                doubling the tree work, and it is pure waste when nothing reads the result.
        """
        started = time.perf_counter()
        row, reasons = self.build_vector(transaction, tier1, tier2, tier3)
        margin = self.margins(row.reshape(1, -1))
        probability = float(self.calibrator.apply(margin)[0])
        return MetaResult(
            probability=probability,
            decision=self.decide(probability),
            top_features=tuple(self.explain(row, top_k=3)) if explain else (),
            model_version=self.model_id,
            latency_ms=(time.perf_counter() - started) * 1_000.0,
            degraded=bool(reasons),
            degraded_reason="; ".join(sorted(set(reasons))) or None,
        )

    def save(self, directory: Path) -> Path:
        """Write the booster and its sidecar, returning the booster path.

        Both paths go through :func:`app.ml.registry.artifact_path` rather than being built by
        concatenation, and the booster is written as XGBoost's own JSON rather than pickled.
        """
        directory.mkdir(parents=True, exist_ok=True)
        booster_path = artifact_path(self.model_id, directory, ".json")
        self.booster.save_model(str(booster_path))
        sidecar = {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "spec": self.spec.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "review_threshold": self.review_threshold,
            "block_threshold": self.block_threshold,
            "best_iteration": self.best_iteration,
            "hyperparameters": self.hyperparameters,
        }
        artifact_path(self.model_id, directory, ".meta.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )
        return booster_path

    @classmethod
    def load(cls, model_id: str, directory: Path) -> "MetaModel":
        """Rebuild a model from the artefact directory."""
        import xgboost as xgb

        sidecar = json.loads(
            artifact_path(model_id, directory, ".meta.json").read_text(encoding="utf-8")
        )
        booster = xgb.Booster()
        booster.load_model(str(artifact_path(model_id, directory, ".json")))
        return cls(
            model_id=sidecar["model_id"],
            spec=MetaInputSpec.from_dict(sidecar["spec"]),
            booster=booster,
            calibrator=SigmoidCalibrator.from_dict(sidecar["calibrator"]),
            review_threshold=float(sidecar["review_threshold"]),
            block_threshold=float(sidecar["block_threshold"]),
            best_iteration=(
                None if sidecar.get("best_iteration") is None else int(sidecar["best_iteration"])
            ),
            algorithm=sidecar["algorithm"],
            hyperparameters=dict(sidecar.get("hyperparameters", {})),
        )


def fit_booster(
    matrix: npt.NDArray[np.float64],
    labels: npt.NDArray[np.bool_],
    *,
    feature_names: Sequence[str],
    rounds: int,
    validation: tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]] | None = None,
    early_stopping: int | None = None,
    max_depth: int | None = None,
) -> Any:
    """Fit one XGBoost booster.

    Args:
        matrix: The fitting matrix.
        labels: Binary targets.
        feature_names: Column names, carried so SHAP output is nameable.
        rounds: Boosting rounds.
        validation: Optional early-stopping set. Must never be the split the result is
            reported on.
        early_stopping: Rounds without improvement before stopping.
        max_depth: Overrides the default depth, for the equal-capacity ablation variants.
    """
    import xgboost as xgb

    params = dict(XGBOOST_PARAMS)
    if max_depth is not None:
        params["max_depth"] = max_depth

    train_set = xgb.DMatrix(matrix, label=labels.astype("int8"), feature_names=list(feature_names))
    watchlist = []
    if validation is not None:
        validation_matrix, validation_labels = validation
        watchlist = [
            (
                xgb.DMatrix(
                    validation_matrix,
                    label=validation_labels.astype("int8"),
                    feature_names=list(feature_names),
                ),
                "val",
            )
        ]
    return xgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping if watchlist else None,
        verbose_eval=False,
    )


def fit_calibrator(
    booster: Any,
    matrix: npt.NDArray[np.float64],
    labels: npt.NDArray[np.bool_],
    *,
    feature_names: Sequence[str],
    iteration_range: tuple[int, int] = (0, 0),
) -> SigmoidCalibrator:
    """Fit Platt scaling on held-out rows, leaving the booster untouched.

    ``FrozenEstimator`` rather than ``cv="prefit"``: the latter was removed in scikit-learn 1.9
    and now raises. Wrapping the booster keeps ``CalibratedClassifierCV`` from refitting it,
    which is the whole point -- the booster was fitted on train and must not see these rows.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    wrapper = _BoosterAsClassifier(booster, tuple(feature_names), iteration_range)
    calibrated = CalibratedClassifierCV(FrozenEstimator(wrapper), method="sigmoid")
    calibrated.fit(matrix, labels.astype("int8"))
    return SigmoidCalibrator.from_sklearn(calibrated)


class _BoosterAsClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Minimal scikit-learn classifier interface over an already-fitted booster.

    Exists only so ``CalibratedClassifierCV`` can consume a raw ``xgboost.Booster``.
    Deliberately not ``XGBClassifier``: re-wrapping the booster in one risks a silent refit, and
    this class cannot refit because :meth:`fit` does nothing.

    ``__sklearn_is_fitted__`` is the documented hook that lets ``FrozenEstimator`` accept an
    object whose fitted state came from outside scikit-learn entirely -- without it,
    ``check_is_fitted`` rejects the wrapper before calibration begins.
    """

    def __init__(
        self,
        booster: Any = None,
        feature_names: tuple[str, ...] = (),
        iteration_range: tuple[int, int] = (0, 0),
    ) -> None:
        self.booster = booster
        self.feature_names = feature_names
        self.iteration_range = iteration_range

    def __sklearn_is_fitted__(self) -> bool:
        """Report the wrapped booster as already fitted. It is; it was fitted on train."""
        return True

    @property
    def classes_(self) -> npt.NDArray[np.int64]:
        """Return the binary class labels scikit-learn expects to find."""
        return np.array([0, 1], dtype="int64")

    def fit(
        self, matrix: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]
    ) -> "_BoosterAsClassifier":
        """Do nothing, and return self.

        Reached only through ``FrozenEstimator``, whose contract is that fitting is a no-op. If
        this ever actually trained something it would refit the booster on the calibration rows,
        which are held out precisely so that cannot happen.
        """
        return self

    def decision_function(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return raw margins (log-odds).

        This method is why the calibration is correct, and its absence is a real bug this
        project shipped once. ``CalibratedClassifierCV`` prefers ``decision_function`` and falls
        back to ``predict_proba``; with only the latter available it fits ``(a, b)`` against
        values in ``[0, 1]``, while :meth:`MetaModel.score_frame` applies them to log-odds. The
        two scales disagree, so the sigmoid is evaluated far outside the range it was fitted on
        and every calibrated probability collapses toward zero.

        The symptom is a giveaway once you know it: a margin-scale Platt fit has ``|a|`` of
        order 1, and the broken fit produced ``a = -97.5``, which is the slope you get fitting a
        sigmoid across a probability band a fraction of a unit wide.
        """
        import xgboost as xgb

        return np.asarray(
            self.booster.predict(
                xgb.DMatrix(matrix, feature_names=list(self.feature_names)),
                output_margin=True,
                iteration_range=self.iteration_range,
            ),
            dtype="float64",
        )

    def predict_proba(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return two-column class probabilities, as scikit-learn expects."""
        import xgboost as xgb

        positive = np.asarray(
            self.booster.predict(
                xgb.DMatrix(matrix, feature_names=list(self.feature_names)),
                iteration_range=self.iteration_range,
            ),
            dtype="float64",
        )
        return np.column_stack([1.0 - positive, positive])

    def predict(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Return hard class labels at 0.5. Present only to satisfy the interface."""
        return (self.predict_proba(matrix)[:, 1] >= 0.5).astype("int64")


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability of the calibrated probabilities, as numbers rather than only a picture.

    ``notebooks/README.md`` is explicit that a number existing only inside a plot is not a
    result, so the bin table travels into the registry alongside the PNG.
    """

    bins: tuple[dict[str, float], ...]
    brier: float
    expected_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form."""
        return {
            "brier": round(self.brier, 6),
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "bins": [dict(row) for row in self.bins],
        }


def calibration_curve_points(
    labels: npt.NDArray[np.bool_],
    probabilities: npt.NDArray[np.float64],
    bins: int = 10,
) -> CalibrationReport:
    """Bin predicted probability against observed frequency.

    Equal-width bins rather than equal-count. On a 3.5% base rate almost every prediction sits
    near zero, and equal-count bins would put nine of ten boundaries inside that spike and say
    nothing about the high-probability region, which is the region a threshold actually lives in.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)

    rows: list[dict[str, float]] = []
    absolute_gap = 0.0
    for position in range(bins):
        mask = index == position
        count = int(mask.sum())
        if count == 0:
            continue
        predicted = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        rows.append(
            {
                "bin_lower": float(edges[position]),
                "bin_upper": float(edges[position + 1]),
                "count": count,
                "mean_predicted": round(predicted, 6),
                "observed_frequency": round(observed, 6),
            }
        )
        absolute_gap += abs(predicted - observed) * count

    brier = float(np.mean((probabilities - labels.astype("float64")) ** 2))
    ece = absolute_gap / len(probabilities) if len(probabilities) else 0.0
    return CalibrationReport(bins=tuple(rows), brier=brier, expected_calibration_error=float(ece))


def benchmark_latency(
    model: MetaModel,
    samples: Sequence[
        tuple[TransactionFeatures, Tier1Result | None, Tier2Result | None, Tier3Result | None]
    ],
) -> dict[str, float]:
    """Time ``predict`` over prepared samples, returning the percentile summary."""
    timings: list[float] = []
    for transaction, tier1, tier2, tier3 in samples:
        started = time.perf_counter()
        model.predict(transaction, tier1, tier2, tier3)
        timings.append((time.perf_counter() - started) * 1_000.0)
    if not timings:
        return {"calls": 0.0}
    ordered = np.sort(np.asarray(timings, dtype="float64"))
    return {
        "calls": float(len(ordered)),
        "p50_ms": round(float(np.percentile(ordered, 50)), 5),
        "p95_ms": round(float(np.percentile(ordered, 95)), 5),
        "p99_ms": round(float(np.percentile(ordered, 99)), 5),
        "mean_ms": round(float(ordered.mean()), 5),
        "max_ms": round(float(ordered.max()), 5),
    }


def build_spec(
    blocks: Sequence[str],
    *,
    tier_model_versions: Mapping[str, str],
    upstream_feature_versions: Mapping[str, str],
    tier3_prefix: str = "tier3_",
) -> MetaInputSpec:
    """Build a spec for the retained blocks, with Tier-3 names pointed at the right variant."""
    from app.models.meta_features import _prefixed

    names = _prefixed(block_feature_names(blocks), tier3_prefix)
    return MetaInputSpec(
        source_dataset="ieee_cis",
        blocks=tuple(sorted(blocks)),
        feature_names=names,
        tier_model_versions=dict(tier_model_versions),
        upstream_feature_versions=dict(upstream_feature_versions),
        tier3_prefix=tier3_prefix,
    )
