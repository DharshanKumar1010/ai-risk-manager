"""Tier-1: real-time per-transaction anomaly scoring.

An Isolation Forest unsupervised baseline against a LightGBM supervised classifier, compared
on held-out PR-AUC, with the winner exposed as::

    score(transaction: TransactionFeatures) -> Tier1Result

Tier-1 is scoreable at ingestion with no dependency on Tier-2 or Tier-3. What that permits is
worked out in :mod:`app.models.tier1_features`; training and the comparison live in
:mod:`app.models.train_tier1`.

**What ``score()`` expects, and the Phase 7 consequence.** ``TransactionFeatures.features``
is an *already-assembled* vector — this module consumes one, it does not build one. The vector
it needs is the Tier-1 feature set, not the 23-key Phase 1 pipeline set stored in
``transactions.features``: Tier-1 also reads the raw row columns the pipeline deliberately
kept out of JSONB. So the scoring path in Phase 7 assembles from two places —

- **raw row columns** (``C1``-``C14``, ``D1``-``D15``, card/address/identity) come from the
  arriving payload itself, at no lookup cost;
- **account-state features** (``velocity_*``, the z-score, the familiarity flags) come from one
  indexed range scan on ``ix_transactions_account_time``;
- **frequency encodings** are an O(1) dict lookup against tables shipped inside the model.

— tags the result with the Tier-1 ``feature_version``, and calls this. The latency budget below
covers the scoring call only; the assembly cost is Phase 7's to measure and is recorded
separately in ``BUILD_LOG.md``.

**A missing feature is an error, never a zero.** :meth:`Tier1Model.score` raises when the
vector's keys or its ``feature_version`` do not match what the model was fitted on. Silently
zero-filling would produce a wrong decision underneath a correct-looking audit row, which is
the one failure mode an audit trail is supposed to make impossible.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.data.schema import TransactionFeatures
from app.ml.registry import artifact_path
from app.models.tier1_features import Tier1InputSpec

#: Latency budget for one scoring call, from PHASE_PROMPTS.md Phase 2.
LATENCY_BUDGET_P95_MS = 50.0

#: Calls in the benchmark, per the phase brief.
LATENCY_BENCHMARK_CALLS = 100

Algorithm = Literal["isolation_forest", "lightgbm"]


class Tier1Result(BaseModel):
    """One Tier-1 decision, as returned to the caller and recorded by the audit trail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Higher means more suspicious. Comparable only within one model_version.",
    )
    is_anomaly: bool = Field(
        description="score >= the model's operating threshold, which was chosen on the "
        "validation split by minimising estimated cost.",
    )
    latency_ms: float = Field(ge=0.0, description="Wall-clock time for this scoring call.")
    model_version: str = Field(
        min_length=1,
        description="Registry model_id. Resolves through models/registry.json to the exact "
        "algorithm, feature definition and held-out numbers this decision rests on.",
    )


@dataclass(frozen=True)
class ScoreNormaliser:
    """Maps a model's native output onto ``[0, 1]``.

    LightGBM's binary objective already returns a probability, so its normaliser is the
    identity. ``IsolationForest.score_samples`` returns an unbounded negative quantity where
    *higher means more normal* — the opposite direction — so its raw output is negated during
    scoring and then min-max mapped using bounds fitted on train.

    The map is monotone, so PR-AUC and every other rank-based metric are unchanged by it. It
    exists so that thresholds, stored scores and dashboard values all live in one space
    regardless of which algorithm won.
    """

    kind: Literal["identity", "minmax"]
    low: float = 0.0
    high: float = 1.0

    @classmethod
    def fit(
        cls, kind: Literal["identity", "minmax"], raw: npt.NDArray[np.float64]
    ) -> "ScoreNormaliser":
        """Fit the map on training scores."""
        if kind == "identity":
            return cls(kind="identity")
        low, high = float(np.min(raw)), float(np.max(raw))
        # A degenerate range would divide by zero. Widening it maps everything to 0.5, which
        # is the honest reading of "this model gave every row the same score".
        if high <= low:
            high = low + 1.0
        return cls(kind="minmax", low=low, high=high)

    def apply(self, raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map raw scores into ``[0, 1]``, clipping values beyond the fitted range."""
        if self.kind == "identity":
            return np.clip(raw, 0.0, 1.0)
        return np.clip((raw - self.low) / (self.high - self.low), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape for the model sidecar."""
        return {"kind": self.kind, "low": self.low, "high": self.high}


@dataclass
class Tier1Model:
    """A fitted Tier-1 scorer plus everything needed to reproduce and audit its decisions.

    Attributes:
        model_id: Registry id, reported as ``Tier1Result.model_version``.
        algorithm: Which of the two candidates this is.
        spec: The fitted input definition.
        threshold: Operating point in normalised score space, chosen on validation by cost.
        normaliser: Native output to ``[0, 1]``.
        estimator: The fitted LightGBM ``Booster`` or ``IsolationForest``.
        numeric_only: True when the estimator was fitted on the numeric-only matrix.
    """

    model_id: str
    algorithm: Algorithm
    spec: Tier1InputSpec
    threshold: float
    normaliser: ScoreNormaliser
    estimator: Any
    numeric_only: bool
    _codes: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    _required: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        """Precompute the lookups the serving path uses on every call.

        Built once rather than per call because they are what makes scoring fast: assembling a
        one-row ``DataFrame`` with 28 ``pd.Categorical`` columns costs ~18ms, and handing
        LightGBM a pandas frame costs a further ~9ms while it re-derives the category mapping.
        The same prediction from a plain float array takes ~0.05ms.
        """
        self._codes = {
            column: {level: index for index, level in enumerate(levels)}
            for column, levels in self.spec.categories.items()
        }
        self._required = frozenset(self.feature_names)

    @property
    def feature_version(self) -> str:
        """Return the Tier-1 feature version this model was fitted against."""
        return self.spec.to_feature_definition().feature_version

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the feature names this model's matrix carries, in order."""
        return self.spec.numeric_feature_names if self.numeric_only else self.spec.feature_names

    def _raw_scores(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return raw suspicion scores for a prepared array, higher meaning more suspicious."""
        if self.algorithm == "lightgbm":
            return np.asarray(self.estimator.predict(matrix), dtype="float64")
        # score_samples is higher-is-more-normal; negate so the direction matches every other
        # score in this project. Getting this backwards yields a PR-AUC below the base-rate
        # floor, which reads as "the model failed" rather than as the sign bug it is.
        return -np.asarray(self.estimator.score_samples(matrix), dtype="float64")

    def _matrix_to_array(self, matrix: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Convert a prepared matrix to the float array both estimators are scored on.

        Categorical columns become their integer codes. That is exactly the representation
        LightGBM built internally from the pandas categoricals at training time, so predictions
        through this path are identical to predictions on the frame itself — asserted by
        ``tests/test_tier1.py::test_numpy_and_pandas_paths_agree`` rather than assumed, because
        the equivalence rests on the level ordering staying in step.
        """
        array = np.empty((len(matrix), len(self.feature_names)), dtype="float64")
        for position, name in enumerate(self.feature_names):
            column = matrix[name]
            array[:, position] = (
                column.cat.codes.to_numpy(dtype="float64")
                if isinstance(column.dtype, pd.CategoricalDtype)
                else column.to_numpy(dtype="float64")
            )
        return array

    def prepare(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Build the scoring array for a whole split."""
        matrix = (
            self.spec.transform_numeric(frame) if self.numeric_only else self.spec.transform(frame)
        )
        return self._matrix_to_array(matrix)

    def score_frame(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Score a whole split. Used by evaluation and by the training run, not by serving."""
        return self.normaliser.apply(self._raw_scores(self.prepare(frame)))

    def _vector_to_array(self, vector: Mapping[str, Any]) -> npt.NDArray[np.float64]:
        """Arrange one assembled feature vector into this model's row, in column order.

        No frequency encoding happens here — the assembled vector already carries the
        ``freq_*`` values, because they are part of the Tier-1 feature set the caller built.

        An unrecognised categorical level becomes ``-1``, which is how LightGBM encodes a
        missing category. That is the right reading: a device or card type never seen in
        training is genuinely unknown to this model, and pretending it is some other level
        would be worse than admitting it.
        """
        row = np.empty((1, len(self.feature_names)), dtype="float64")
        for position, name in enumerate(self.feature_names):
            value = vector[name]
            codes = self._codes.get(name)
            if codes is not None:
                row[0, position] = -1.0 if value is None else codes.get(str(value), -1)
            elif value is None:
                row[0, position] = np.nan
            else:
                row[0, position] = float(value)
        return row

    def top_feature_importances(self, count: int = 12) -> list[tuple[str, float]]:
        """Return the features carrying the most split gain, as shares of the total.

        Reported with every result because a headline metric does not say what produced it.
        A model leaning almost entirely on one or two features is the signature of a dataset
        artefact rather than of a learned pattern, and that is worth seeing before the number
        is believed — ml-evaluation-standards section 4.

        Note the limit: gain is per-feature, so it cannot reveal a *relation between* two
        features. PaySim's fraud tell is ``amount == oldbalanceOrg``, which shows up here only
        as two moderately important features rather than as the near-deterministic rule it is.

        Returns:
            ``(feature_name, share_of_total_gain)``, largest first. Empty for Isolation
            Forest, which has no comparable notion.
        """
        if self.algorithm != "lightgbm":
            return []
        gains = np.asarray(
            self.estimator.feature_importance(importance_type="gain"), dtype="float64"
        )
        total = float(gains.sum())
        if total <= 0:
            return []
        ranked = sorted(
            zip(self.feature_names, gains / total, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(name, float(share)) for name, share in ranked[:count]]

    def _require_compatible(self, transaction: TransactionFeatures) -> None:
        """Raise unless the vector matches what this model was fitted on.

        Two separate checks because they fail differently. A ``feature_version`` mismatch means
        the caller assembled against a different definition — the whole vector is suspect. A
        missing key means this one assembly is incomplete.
        """
        if transaction.feature_version != self.feature_version:
            raise ValueError(
                f"Transaction {transaction.transaction_id!r} carries feature_version "
                f"{transaction.feature_version!r}, but model {self.model_id} was fitted on "
                f"{self.feature_version!r}. Refusing to score: the vectors are not the same "
                "definition, and scoring anyway would attach a wrong decision to a "
                "correct-looking audit row."
            )
        if not self._required.issubset(transaction.features):
            missing = sorted(self._required.difference(transaction.features))
            raise ValueError(
                f"Transaction {transaction.transaction_id!r} is missing "
                f"{len(missing)} Tier-1 feature(s): {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}. A missing feature is an error, not a "
                "zero — zero-filling would silently change the decision."
            )

    def score(self, transaction: TransactionFeatures) -> Tier1Result:
        """Score one transaction.

        Args:
            transaction: Carries the assembled Tier-1 feature vector and its version.

        Returns:
            The decision, its normalised score, the measured latency and the model version.

        Raises:
            ValueError: If the vector's ``feature_version`` or key set does not match.
        """
        started = time.perf_counter()
        self._require_compatible(transaction)
        row = self._vector_to_array(transaction.features)
        score = float(self.normaliser.apply(self._raw_scores(row))[0])
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return Tier1Result(
            score=score,
            is_anomaly=score >= self.threshold,
            latency_ms=elapsed_ms,
            model_version=self.model_id,
        )

    # --- Persistence ---------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Write the model and its sidecar under ``directory``.

        LightGBM is saved in its **native text format** rather than pickled. Loading a pickle
        executes arbitrary code, and this load path becomes reachable from a Phase 7 endpoint;
        the text format has no such property. Isolation Forest has no text format, so it goes
        through joblib and is loaded only from the configured artefact directory — never from
        a caller-supplied path.
        """
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        if self.algorithm == "lightgbm":
            artifact = artifact_path(self.model_id, directory, ".txt")
            self.estimator.save_model(str(artifact))
        else:
            artifact = artifact_path(self.model_id, directory, ".joblib")
            joblib.dump(self.estimator, artifact)

        sidecar = artifact_path(self.model_id, directory, ".meta.json")
        sidecar.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "algorithm": self.algorithm,
                    "threshold": self.threshold,
                    "numeric_only": self.numeric_only,
                    "normaliser": self.normaliser.to_dict(),
                    "feature_version": self.feature_version,
                    "feature_names": list(self.feature_names),
                    "spec": {
                        "source_dataset": self.spec.source_dataset,
                        "numeric_columns": list(self.spec.numeric_columns),
                        "native_categorical_columns": list(self.spec.native_categorical_columns),
                        "frequency_columns": list(self.spec.frequency_columns),
                        "categories": {k: list(v) for k, v in self.spec.categories.items()},
                        "encoders": {k: dict(v) for k, v in self.spec.encoders.items()},
                        "post_settlement_columns": list(self.spec.post_settlement_columns),
                        "dropped": [str(item) for item in self.spec.dropped],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return artifact

    @classmethod
    def load(
        cls,
        model_id: str,
        directory: Path,
        feature_version: str | None = None,
    ) -> "Tier1Model":
        """Rebuild a model from its artefact and sidecar.

        The inverse of :meth:`save`. Phase 2 only needed ``save``, so the reconstruction lived
        inline in ``meta_features.load_registered_tier1``; Phase 7 serves from it on every
        request, so it belongs on the type.

        **Only the LightGBM path is loadable.** Isolation Forest persists through joblib, and
        unpickling executes arbitrary code — this method is reachable from an HTTP endpoint, so
        it refuses that format outright rather than gating it on a flag someone could pass.
        The unsupervised model stays available to offline analysis, which loads it deliberately
        and not from a request path.

        Args:
            model_id: Bare registry identifier. Becomes a path only through
                :func:`app.ml.registry.artifact_path`, which proves it cannot escape
                ``directory`` — it arrives from ``registry.json``, which is file content and
                therefore not a trusted identifier at the point it becomes a path.
            directory: The artefact directory.
            feature_version: When given, the hash the registry records for this model. The
                rebuilt spec is required to hash to it.

        Returns:
            The reconstructed model.

        Raises:
            FileNotFoundError: If the sidecar or the booster is absent.
            ValueError: If the sidecar describes a model this path refuses to load.
            RuntimeError: If the rebuilt spec does not hash to ``feature_version``. Every
                provenance claim downstream of this object rests on that hash, so a mismatch
                is refused rather than recorded.
        """
        import lightgbm as lgb

        sidecar_path = artifact_path(model_id, directory, ".meta.json")
        booster_path = artifact_path(model_id, directory, ".txt")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Tier-1 sidecar not found: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        algorithm = sidecar["algorithm"]
        if algorithm != "lightgbm":
            raise ValueError(
                f"Tier1Model.load refuses algorithm {algorithm!r}. Only the native LightGBM "
                "text format is loadable here: the other candidate persists through joblib, "
                "and this path is reachable from a scoring endpoint."
            )
        if not booster_path.exists():
            raise FileNotFoundError(f"Tier-1 booster not found: {booster_path}")

        spec = Tier1InputSpec.from_dict(sidecar["spec"])
        rebuilt = spec.to_feature_definition().feature_version
        if feature_version is not None and rebuilt != feature_version:
            raise RuntimeError(
                f"rebuilt Tier-1 spec hashes to {rebuilt} but {feature_version} was expected "
                f"for {model_id}. The reconstruction has drifted from what fit_tier1_input_spec "
                "produces, so any feature_version recorded from it would be unresolvable."
            )

        normaliser = sidecar["normaliser"]
        return cls(
            model_id=sidecar["model_id"],
            algorithm=algorithm,
            spec=spec,
            threshold=float(sidecar["threshold"]),
            normaliser=ScoreNormaliser(
                kind=normaliser["kind"],
                low=float(normaliser["low"]),
                high=float(normaliser["high"]),
            ),
            estimator=lgb.Booster(model_file=str(booster_path)),
            numeric_only=bool(sidecar["numeric_only"]),
        )


def benchmark_latency(
    model: Tier1Model,
    transactions: Sequence[TransactionFeatures],
    calls: int = LATENCY_BENCHMARK_CALLS,
) -> dict[str, float]:
    """Time sequential scoring calls and return the latency percentiles.

    Sequential and one-at-a-time on purpose: it is the serving shape. A batched benchmark
    would report a throughput number dressed up as a latency one.

    Returns:
        ``p50``/``p95``/``p99``/``mean``/``max`` in milliseconds, plus the call count.
    """
    if not transactions:
        raise ValueError("Latency benchmark needs at least one transaction")
    timings: list[float] = []
    for index in range(calls):
        result = model.score(transactions[index % len(transactions)])
        timings.append(result.latency_ms)
    measured = np.array(timings)
    return {
        "calls": float(calls),
        "p50_ms": float(np.percentile(measured, 50)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
        "mean_ms": float(np.mean(measured)),
        "max_ms": float(np.max(measured)),
    }


def explain(
    model: Tier1Model,
    transaction: TransactionFeatures,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the top-k signed SHAP contributions behind one Tier-1 score.

    Tree SHAP over an already-trained booster, so this costs a single extra traversal. Phase 5
    fuses these into the meta-learner's explanation and Phase 8 renders them; the audit trail
    wants them from now rather than later.

    Only meaningful for the LightGBM model — Isolation Forest's path-length score has no
    additive attribution of this kind.

    Returns:
        ``(feature_name, contribution)`` pairs, largest absolute contribution first. A positive
        contribution pushed the score up, towards "suspicious".
    """
    if model.algorithm != "lightgbm":
        raise ValueError(
            "SHAP attribution is defined here only for the LightGBM model; Isolation Forest's "
            "path-length score is not additively decomposable in the same way."
        )
    model._require_compatible(transaction)
    row = model._vector_to_array(transaction.features)
    # pred_contrib returns one column per feature plus a trailing base value.
    contributions = np.asarray(model.estimator.predict(row, pred_contrib=True), dtype="float64")[0][
        :-1
    ]
    ranked = sorted(
        zip(model.feature_names, contributions, strict=True),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    return [(name, float(value)) for name, value in ranked[:top_k]]


class OnlineRecalibrator(Protocol):
    """Interface stub for future online recalibration — **designed for, not implemented.**

    Fraud models drift: the base rate moves, attack patterns rotate, and a threshold chosen on
    one validation window stops being the cost-minimising one. The intended shape is that
    confirmed outcomes (chargebacks, manual review dispositions) arrive after the fact and
    shift the operating threshold without refitting the model.

    Deliberately not built in Phase 2. Recalibrating against live outcomes needs a labelled
    feedback path that does not exist in this project -- Phase 9's webhook scores live
    transactions but records only this service's own allow/review/block decision in
    ``audit_log``, never a confirmed outcome, so the labelled path this class needs still does
    not exist (Phase 9.5 corrected this docstring, which had promised Phase 9 would supply it).
    Building this now would mean building it against imagined data. Recorded in
    ``BUILD_LOG.md`` as designed-for.
    """

    def observe(self, model_version: str, score: float, was_fraud: bool) -> None:
        """Record one confirmed outcome for a past decision."""
        ...

    def recommended_threshold(self, model_version: str) -> float | None:
        """Return an updated operating threshold, or None while evidence is insufficient."""
        ...
