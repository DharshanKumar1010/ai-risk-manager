"""Train, ablate and register the RiskIQ meta-learner (Phase 5).

**What this phase is actually for.** Not a PR-AUC win. Phase 2 measured Tier-1 at 0.5276 on the
held-out test split; Phase 3 measured Tier-2 losing to it head-to-head by a paired delta whose
interval is [-0.5110, -0.4527]; Phase 4 measured Tier-3's per-transaction projection *below*
no-skill and its fusion into Tier-1 significantly negative at -0.0031, CI [-0.0038, -0.0026].
Both phases deferred the same question to here, in the same words: fit the meta-learner with and
without each layer, and keep the layer only if the paired delta excludes zero. This module is
that experiment. A result of "the meta-learner ties Tier-1" is the expected outcome and is
reported as a finding, not padded into a win.

**The measurement design, and why validation is cut three ways.**

Tier-1's scores on the train split are in-sample, so the fitting matrix uses out-of-fold Tier-1
(``meta_features.build_oof_tier1``). That fixes Tier-1. It does not fix Tier-2, which is
contaminated differently and worse: the autoencoder was fitted on fraud-free train windows
selected *by account label*, so on train, fraud rows come from accounts withheld entirely while
clean rows mostly come from accounts inside the fit. The gap that manufactures does not
reproduce at test.

Validation has none of these problems -- no tier was fitted on it -- so the keep/drop verdict is
read from a validation-fitted arbiter, while the shipped model is the train-fitted one. Running
both and reporting both delta columns turns the contamination into a measured quantity: the
disagreement between the columns *is* its size.

One asymmetry has to be stated because it decides how the verdict may be read. Tier-2's latent
size, window and early-stopping round were all selected on validation, so validation flatters
Tier-2 relative to test. That makes a **drop** verdict safe -- the layer failed with a thumb on
the scale -- and a **keep** verdict merely suggestive. The report says so rather than treating
the two directions as equally solid.

**Power.** The arbiter slice carries roughly a third of validation's positives, so its intervals
are appreciably wider than Phase 4's full-test comparison. Under a drop-hard rule an underpowered
arbiter retires layers by default, which is a design bias rather than a finding, so every delta
is reported with its interval width and every retirement is worded as "retired for want of
evidence at this n", never as "measured to add nothing".
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
from app.data.splitting import find_boundary_overlaps
from app.ml.cost import (
    CostEstimate,
    CostModel,
    choose_threshold_by_cost,
    cost_at_threshold,
    render_sensitivity,
    review_cost_sweep,
    sensitivity_sweep,
    threshold_for_flag_rate,
)
from app.ml.evaluation import (
    EvaluationResult,
    bootstrap_pr_auc,
    bootstrap_pr_auc_delta,
    confusion_at_threshold,
    evaluate,
    pr_auc,
)
from app.ml.registry import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REGISTRY_PATH,
    RegistryEntry,
    append_entry,
    build_model_id,
    read_registry,
)
from app.models.meta_features import (
    EXEMPT_BLOCKS,
    OOF_BLOCKS,
    OOF_SCOREABLE_COLUMN,
    RANDOM_SEED,
    Tier2Features,
    Tier3Features,
    assemble_meta_frame,
    available_blocks,
    build_oof_tier1,
    build_tier2_errors,
    build_tier3_features,
    load_registered_tier1,
    matrix_for,
    require_ieee_cis,
    tier2_memorisation_diagnostic,
)
from app.models.meta_learner import (
    ABLATION_MAX_DEPTH,
    ABLATION_NUM_ROUNDS,
    XGBOOST_EARLY_STOPPING,
    XGBOOST_NUM_ROUNDS,
    XGBOOST_PARAMS,
    CalibrationReport,
    MetaModel,
    build_spec,
    calibration_curve_points,
    fit_booster,
    fit_calibrator,
)

logger = logging.getLogger("riskiq.meta")

SOURCE: SourceDataset = "ieee_cis"
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")

IEEE_COST_UNITS = "IEEE-CIS amount units (consistent with USD)"

#: The review queue cannot absorb more than this share of traffic, whatever the cost arithmetic
#: prefers. Same cap Tiers 1-3 report their capacity-constrained operating point at.
MAX_REVIEW_FLAG_RATE = 0.01

#: Precision a decision must reach before it may block outright rather than go to a human.
#: A block declines a real customer, so it is held to a standard a review is not.
MIN_BLOCK_PRECISION = 0.80

#: Minimum flagged rows a candidate block threshold must cover before its precision is believed.
#: Without this the scan picks the highest score in the split, where "precision" is one row.
MIN_BLOCK_FLAGGED = 50

#: Written as the block threshold when no candidate clears the bar. Above any attainable
#: probability, so nothing ever blocks and every flag goes to a human -- the safe direction,
#: since a block declines a real customer and a review only costs analyst time. A finite value
#: rather than infinity because the registry entry has to be valid JSON.
BLOCK_DISABLED = 2.0

#: Chronological cut of the validation split, as fractions of its rows.
#:
#: The arbiter measures a *ranking* difference, which is stable across time, so it takes an
#: early slice. The calibrator and the thresholds are level-sensitive, and the base rate visibly
#: drifts across this corpus, so they take the slice closest to test.
VALIDATION_FRACTIONS: tuple[float, float, float] = (0.40, 0.35, 0.25)

#: The registered upstream models this phase fuses. Pinned rather than resolved by "latest",
#: because Tier-3 has sixteen registry entries and Tier-2 three, and picking the wrong one by
#: accident is a documented Phase 4 hazard.
TIER2_MODEL_ID = "tier2-behavioral-lstm-w15-latent8-ieee-cis-20260823t070529z"


# ==========================================================================================
# Loading and slicing
# ==========================================================================================


def load_splits(
    processed_dir: Path, source: SourceDataset, sample: int | None
) -> dict[str, pd.DataFrame]:
    """Load the three chronological splits, refusing anything but IEEE-CIS.

    Raises:
        FileNotFoundError: If a split parquet is missing.
        RuntimeError: If the splits on disk are not strictly time-ordered.
        ValueError: For any corpus other than ``ieee_cis``.
    """
    require_ieee_cis(source)
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


@dataclass(frozen=True)
class ValidationSlices:
    """Validation cut three ways, each with exactly one job.

    Attributes:
        fit: Fits the arbiter models, and early-stops the shipped model.
        arbiter: Where every keep/drop delta is measured. Nothing is fitted on it.
        late: Calibration and operating thresholds. Closest to test, because both are
            level-sensitive and this corpus's base rate drifts.
    """

    fit: pd.DataFrame
    arbiter: pd.DataFrame
    late: pd.DataFrame

    def describe(self) -> list[dict[str, Any]]:
        """Return the slice table for the report."""
        rows = []
        for name, frame in (("V-fit", self.fit), ("V-arb", self.arbiter), ("V-late", self.late)):
            labels = frame["is_fraud"].to_numpy(dtype=bool)
            rows.append(
                {
                    "slice": name,
                    "rows": len(frame),
                    "positives": int(labels.sum()),
                    "base_rate": round(float(labels.mean()), 6) if len(frame) else 0.0,
                    "first_event": str(frame["event_time"].min()),
                    "last_event": str(frame["event_time"].max()),
                }
            )
        return rows


def validation_slices(
    validation: pd.DataFrame, fractions: tuple[float, float, float] = VALIDATION_FRACTIONS
) -> ValidationSlices:
    """Cut validation chronologically into fit / arbiter / late slices.

    Chronological, never stratified. A stratified cut would mix rows across time and let the
    calibrator see the same period the arbiter measured on.

    Raises:
        ValueError: If any slice would be empty or carry no positives -- either makes the slice
            unable to do its job, and continuing would produce a number that looks fine.
    """
    ordered = validation.sort_values("event_time", kind="stable").reset_index(drop=True)
    rows = len(ordered)
    first = int(rows * fractions[0])
    second = first + int(rows * fractions[1])
    slices = ValidationSlices(
        fit=ordered.iloc[:first].copy(),
        arbiter=ordered.iloc[first:second].copy(),
        late=ordered.iloc[second:].copy(),
    )
    for name, frame in (("fit", slices.fit), ("arbiter", slices.arbiter), ("late", slices.late)):
        if frame.empty:
            raise ValueError(f"validation slice {name!r} is empty at fractions {fractions}")
        if not frame["is_fraud"].astype(bool).any():
            raise ValueError(
                f"validation slice {name!r} carries no positives; it cannot fit a calibrator, "
                "choose a threshold or measure a delta"
            )
    logger.info(
        "validation cut: V-fit %d rows / %d pos, V-arb %d / %d, V-late %d / %d",
        len(slices.fit),
        int(slices.fit["is_fraud"].sum()),
        len(slices.arbiter),
        int(slices.arbiter["is_fraud"].sum()),
        len(slices.late),
        int(slices.late["is_fraud"].sum()),
    )
    return slices


# ==========================================================================================
# The ablation
# ==========================================================================================


@dataclass(frozen=True)
class AblationDelta:
    """One paired PR-AUC comparison between two feature sets, with its power.

    Attributes:
        block: The block being added or removed.
        kind: ``leave_one_out`` or ``add_one``.
        delta: Point estimate, ``with`` minus ``without``.
        interval: 95% percentile interval on the paired bootstrap.
        rows: Rows the comparison was measured on.
        positives: Positives among them.
    """

    block: str
    kind: str
    delta: float
    interval: tuple[float, float]
    rows: int
    positives: int

    @property
    def width(self) -> float:
        """Return the interval width -- the honest read on what this test could detect."""
        return self.interval[1] - self.interval[0]

    @property
    def excludes_zero_positively(self) -> bool:
        """Whether the lower bound is strictly above zero."""
        return self.interval[0] > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form."""
        return {
            "block": self.block,
            "kind": self.kind,
            "delta": round(self.delta, 6),
            "interval": [round(self.interval[0], 6), round(self.interval[1], 6)],
            "interval_width": round(self.width, 6),
            "rows": self.rows,
            "positives": self.positives,
            "excludes_zero_positively": self.excludes_zero_positively,
        }


@dataclass(frozen=True)
class BlockVerdict:
    """The keep/drop decision for one block, with the evidence behind it."""

    block: str
    retained: bool
    exempt: bool
    leave_one_out: AblationDelta | None
    add_one: AblationDelta | None

    def rationale(self) -> str:
        """Return the sentence that goes into the registry note.

        A retirement on an interval straddling zero is worded as an absence of evidence, not as
        evidence of absence. The distinction is the difference between an honest claim and a
        false one, and at this sample size it is the usual case.
        """
        if self.exempt:
            return f"{self.block}: retained (base block, exempt from retirement)."
        evidence = [d for d in (self.leave_one_out, self.add_one) if d is not None]
        if not evidence:
            return f"{self.block}: retired -- unavailable, no comparison was possible."
        best = max(evidence, key=lambda d: d.interval[0])
        span = f"[{best.interval[0]:+.4f}, {best.interval[1]:+.4f}]"
        if self.retained:
            return (
                f"{self.block}: retained -- {best.kind} delta {best.delta:+.4f}, 95% CI {span}, "
                f"lower bound above zero at n={best.rows:,} ({best.positives:,} positives)."
            )
        # A block can fail the keep rule two very different ways, and saying "no evidence" for
        # both would be a false statement about the second. An interval lying entirely below
        # zero is not an absence of evidence -- it is evidence the block makes the model worse,
        # which is a stronger and more useful finding than a tie.
        harmful = [d for d in evidence if d.interval[1] < 0.0]
        if harmful:
            worst = min(harmful, key=lambda d: d.interval[1])
            return (
                f"{self.block}: retired -- {worst.kind} delta {worst.delta:+.4f}, 95% CI "
                f"[{worst.interval[0]:+.4f}, {worst.interval[1]:+.4f}], which excludes zero on "
                f"the NEGATIVE side at n={worst.rows:,} ({worst.positives:,} positives). This "
                "block measurably degrades the model; removing it is an improvement, not merely "
                "a simplification."
            )
        return (
            f"{self.block}: retired for want of evidence at n={best.rows:,} "
            f"({best.positives:,} positives) -- best {best.kind} delta {best.delta:+.4f}, "
            f"95% CI {span}, width {best.width:.4f}. The interval does not exclude zero, which "
            "is an absence of evidence that this layer helps, not evidence that it does not."
        )


def _fit_variant(
    fit_frame: pd.DataFrame,
    blocks: Sequence[str],
    prefix: str,
) -> tuple[Any, tuple[str, ...]]:
    """Fit one equal-capacity ablation variant."""
    matrix, names = matrix_for(fit_frame, blocks, prefix)
    labels = fit_frame["is_fraud"].to_numpy(dtype=bool)
    booster = fit_booster(
        matrix,
        labels,
        feature_names=names,
        rounds=ABLATION_NUM_ROUNDS,
        max_depth=ABLATION_MAX_DEPTH,
    )
    return booster, names


def _score_variant(
    booster: Any,
    frame: pd.DataFrame,
    names: Sequence[str],
    iteration_range: tuple[int, int] = (0, 0),
) -> npt.NDArray[np.float64]:
    """Score a frame with an ablation variant, in margin space.

    ``iteration_range`` defaults to every tree, which is right for the ablation variants: they
    are fitted at a fixed round count with no early stopping, precisely so the comparison
    isolates the feature blocks. The shipped model passes its selected range explicitly.
    """
    import xgboost as xgb

    matrix = frame.loc[:, list(names)].to_numpy(dtype="float64")
    return np.asarray(
        booster.predict(
            xgb.DMatrix(matrix, feature_names=list(names)),
            output_margin=True,
            iteration_range=iteration_range,
        ),
        dtype="float64",
    )


@dataclass
class AblationReport:
    """Every variant, every delta and the resulting feature set."""

    variants: dict[str, float] = field(default_factory=dict)
    deltas: list[AblationDelta] = field(default_factory=list)
    verdicts: list[BlockVerdict] = field(default_factory=list)
    retained: tuple[str, ...] = ()
    arbiter_rows: int = 0
    arbiter_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form for the registry."""
        return {
            "arbiter_rows": self.arbiter_rows,
            "arbiter_positives": self.arbiter_positives,
            "variant_pr_auc": {k: round(v, 6) for k, v in self.variants.items()},
            "deltas": [d.to_dict() for d in self.deltas],
            "verdicts": [
                {"block": v.block, "retained": v.retained, "rationale": v.rationale()}
                for v in self.verdicts
            ],
            "retained_blocks": list(self.retained),
        }


def run_ablation(
    fit_frame: pd.DataFrame,
    arbiter_frame: pd.DataFrame,
    blocks: Sequence[str],
    *,
    prefix: str = "tier3_",
    label: str = "validation-fit",
) -> AblationReport:
    """Fit every ablation variant and measure each block's paired contribution.

    Variants are fitted on ``fit_frame`` and compared on ``arbiter_frame``, which nothing is
    fitted on. All variants use the same rounds and depth so the comparison isolates the feature
    blocks rather than how much fitting each got.

    Both directions are measured. Leave-one-out alone can retire two mutually redundant blocks
    that are jointly useful; add-one alone can miss a block that only helps in combination.
    """
    labels = arbiter_frame["is_fraud"].to_numpy(dtype=bool)
    report = AblationReport(arbiter_rows=len(arbiter_frame), arbiter_positives=int(labels.sum()))

    full_booster, full_names = _fit_variant(fit_frame, blocks, prefix)
    full_scores = _score_variant(full_booster, arbiter_frame, full_names)
    report.variants["full"] = pr_auc(labels, full_scores)
    logger.info("[%s] full variant: arbiter PR-AUC %.4f", label, report.variants["full"])

    base = [b for b in blocks if b in EXEMPT_BLOCKS]
    scores_by_set: dict[tuple[str, ...], npt.NDArray[np.float64]] = {
        tuple(sorted(blocks)): full_scores
    }

    def scores_for(subset: Sequence[str]) -> npt.NDArray[np.float64]:
        key = tuple(sorted(subset))
        if key not in scores_by_set:
            booster, names = _fit_variant(fit_frame, list(key), prefix)
            scores_by_set[key] = _score_variant(booster, arbiter_frame, names)
            report.variants["+".join(key)] = pr_auc(labels, scores_by_set[key])
        return scores_by_set[key]

    if base:
        report.variants["base"] = pr_auc(labels, scores_for(base))

    for block in blocks:
        exempt = block in EXEMPT_BLOCKS
        without = [b for b in blocks if b != block]
        leave_one_out: AblationDelta | None = None
        add_one: AblationDelta | None = None

        if without:
            interval = bootstrap_pr_auc_delta(
                labels, full_scores, scores_for(without), seed=RANDOM_SEED
            )
            leave_one_out = AblationDelta(
                block=block,
                kind="leave_one_out",
                delta=report.variants["full"] - pr_auc(labels, scores_for(without)),
                interval=interval,
                rows=report.arbiter_rows,
                positives=report.arbiter_positives,
            )
            report.deltas.append(leave_one_out)

        if not exempt and base:
            with_block = [*base, block]
            interval = bootstrap_pr_auc_delta(
                labels, scores_for(with_block), scores_for(base), seed=RANDOM_SEED
            )
            add_one = AblationDelta(
                block=block,
                kind="add_one",
                delta=pr_auc(labels, scores_for(with_block)) - pr_auc(labels, scores_for(base)),
                interval=interval,
                rows=report.arbiter_rows,
                positives=report.arbiter_positives,
            )
            report.deltas.append(add_one)

        retained = exempt or any(
            d is not None and d.excludes_zero_positively for d in (leave_one_out, add_one)
        )
        verdict = BlockVerdict(
            block=block,
            retained=retained,
            exempt=exempt,
            leave_one_out=leave_one_out,
            add_one=add_one,
        )
        report.verdicts.append(verdict)
        logger.info("[%s] %s", label, verdict.rationale())

    report.retained = tuple(v.block for v in report.verdicts if v.retained)
    return report


# ==========================================================================================
# Calibration and thresholds
# ==========================================================================================


@dataclass(frozen=True)
class OperatingPoints:
    """The thresholds, chosen out-of-fold on V-late, with the evidence behind them."""

    review_threshold: float
    block_threshold: float
    review_criterion: str
    block_criterion: str
    cost: CostEstimate
    capacity_threshold: float
    sensitivity: str
    review_sweep: str


def crossfit_calibrated_scores(
    booster: Any,
    frame: pd.DataFrame,
    names: Sequence[str],
    iteration_range: tuple[int, int] = (0, 0),
) -> npt.NDArray[np.float64]:
    """Return out-of-fold calibrated probabilities over V-late.

    Two chronological halves: the calibrator fitted on the first scores the second and vice
    versa. Without this the operating threshold would be chosen on the same rows the calibrator
    was fitted on, which makes it optimistic by exactly the amount the calibrator overfits.
    """
    matrix = frame.loc[:, list(names)].to_numpy(dtype="float64")
    labels = frame["is_fraud"].to_numpy(dtype=bool)
    half = len(frame) // 2
    out = np.empty(len(frame), dtype="float64")

    for fit_slice, score_slice in (
        (slice(0, half), slice(half, None)),
        (slice(half, None), slice(0, half)),
    ):
        if not labels[fit_slice].any():
            # A half with no positives cannot fit a sigmoid; fall back to the whole slice and
            # say so, rather than emitting a silently degenerate calibrator.
            logger.warning("V-late half has no positives; calibrating on the full slice")
            calibrator = fit_calibrator(
                booster, matrix, labels, feature_names=names, iteration_range=iteration_range
            )
            return calibrator.apply(
                np.asarray(_score_variant(booster, frame, names, iteration_range), dtype="float64")
            )
        calibrator = fit_calibrator(
            booster,
            matrix[fit_slice],
            labels[fit_slice],
            feature_names=names,
            iteration_range=iteration_range,
        )
        margins = _score_variant(booster, frame.iloc[score_slice], names, iteration_range)
        out[score_slice] = calibrator.apply(margins)
    return out


def choose_operating_points(
    labels: npt.NDArray[np.bool_],
    probabilities: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    cost_model: CostModel,
) -> OperatingPoints:
    """Choose the review and block thresholds on V-late's out-of-fold probabilities.

    Both are chosen on validation and never on test: choosing an operating point on test would
    make test a validation set and invalidate the headline.
    """
    review_threshold, estimate = choose_threshold_by_cost(
        labels, probabilities, amounts, cost_model
    )
    capacity = threshold_for_flag_rate(probabilities, MAX_REVIEW_FLAG_RATE)
    review_criterion = "minimising estimated cost on the V-late out-of-fold probabilities"
    if review_threshold < capacity:
        review_threshold = capacity
        review_criterion = (
            "minimising estimated cost on V-late, raised to respect a "
            f"{MAX_REVIEW_FLAG_RATE:.1%} review-capacity cap"
        )
        estimate = cost_at_threshold(labels, probabilities, amounts, review_threshold, cost_model)

    # A block declines a real customer outright, so it is held to a precision standard a review
    # queue is not, and it must sit *strictly above* the review threshold. Taking the lowest
    # threshold that clears the precision bar anywhere would routinely land below the review
    # point on a well-separated model, and clamping it back up collapses block onto review --
    # which silently removes the human-review band entirely and turns every flag into a decline.
    candidates = np.unique(probabilities)
    block_threshold = BLOCK_DISABLED
    block_criterion = (
        f"no threshold above the review point reached precision {MIN_BLOCK_PRECISION:.0%} over "
        f"at least {MIN_BLOCK_FLAGGED} flagged rows on V-late. Blocking is disabled and every "
        "flag goes to a human, which is the safe direction: a review costs analyst time, a "
        "block declines a real customer."
    )
    for candidate in candidates[candidates > review_threshold]:
        flagged = probabilities >= candidate
        covered = int(flagged.sum())
        if covered < MIN_BLOCK_FLAGGED:
            # Precision over a handful of rows is noise, and the scan would otherwise settle on
            # the single highest-scoring row and call it 100% precise.
            break
        if float(labels[flagged].mean()) >= MIN_BLOCK_PRECISION:
            block_threshold = float(candidate)
            block_criterion = (
                f"lowest V-late threshold above the review point reaching precision "
                f"{MIN_BLOCK_PRECISION:.0%} over at least {MIN_BLOCK_FLAGGED} flagged rows"
            )
            break

    return OperatingPoints(
        review_threshold=float(review_threshold),
        block_threshold=block_threshold,
        review_criterion=review_criterion,
        block_criterion=block_criterion,
        cost=estimate,
        capacity_threshold=float(capacity),
        sensitivity=render_sensitivity(
            sensitivity_sweep(labels, probabilities, amounts, cost_model),
            "Cost sensitivity (both parameters scaled together)",
        ),
        review_sweep=render_sensitivity(
            review_cost_sweep(labels, probabilities, amounts, cost_model),
            "Review cost sensitivity (false-positive cost alone)",
        ),
    )


# ==========================================================================================
# The run
# ==========================================================================================


@dataclass
class MetaRunReport:
    """Everything one Phase 5 run produced."""

    model: MetaModel
    result: EvaluationResult
    tier1_result: EvaluationResult
    delta_vs_tier1: tuple[float, float]
    delta_point: float
    ablation_validation: AblationReport
    ablation_train: AblationReport | None
    calibration: CalibrationReport
    isotonic_comparison: dict[str, Any]
    operating: OperatingPoints
    slices: list[dict[str, Any]]
    handicap: list[dict[str, Any]]
    tier2_diagnostic: dict[str, Any] | None
    tier2_diagnostic_reading: list[str]
    best_iteration: int | str
    tier3_circularity: dict[str, Any] | None
    latency: dict[str, float]
    matched_flag_rate: dict[str, Any]
    training_window: dict[str, str]
    notes: list[str] = field(default_factory=list)


def _amounts(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return the transaction amounts a cost estimate weights false negatives by."""
    return frame["amount"].to_numpy(dtype="float64")


def _labels(frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Return the binary labels."""
    return frame["is_fraud"].to_numpy(dtype=bool)


def isotonic_comparison(
    booster: Any,
    late: pd.DataFrame,
    names: Sequence[str],
    test: pd.DataFrame,
    sigmoid_probabilities: npt.NDArray[np.float64],
    iteration_range: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Fit isotonic calibration as the documented losing alternative.

    ml-evaluation-standards section 4 requires baselines that lost to be kept as comparisons
    rather than silently discarded. The expected finding -- isotonic calibrates marginally
    better and costs PR-AUC by tying scores -- is itself instructive, because it is the reason
    the shipped calibrator is a sigmoid.
    """
    from sklearn.isotonic import IsotonicRegression

    late_margins = _score_variant(booster, late, names, iteration_range)
    test_margins = _score_variant(booster, test, names, iteration_range)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(late_margins, _labels(late).astype("float64"))
    isotonic_test = np.asarray(isotonic.predict(test_margins), dtype="float64")

    test_labels = _labels(test)
    isotonic_report = calibration_curve_points(test_labels, isotonic_test)
    sigmoid_report = calibration_curve_points(test_labels, sigmoid_probabilities)
    return {
        "verdict": (
            "Not shipped, and 'lost' overstates it. Isotonic is piecewise-constant, so it ties "
            "scores and moves PR-AUC by an artefact of step width; the sigmoid is strictly "
            "monotone and leaves the ranking exactly intact, which is why the sigmoid ships "
            "when PR-AUC is the headline. On calibration itself isotonic is the better of the "
            "two, and by how much is worth reading rather than waving at: compare the two ECE "
            "figures below. A large gap is not a free win for isotonic -- it is a signal worth "
            "investigating, because a correctly-fitted sigmoid on this data should be close."
        ),
        "isotonic": {
            "pr_auc": round(pr_auc(test_labels, isotonic_test), 6),
            "brier": round(isotonic_report.brier, 6),
            "expected_calibration_error": round(isotonic_report.expected_calibration_error, 6),
            "distinct_scores": int(np.unique(isotonic_test).size),
        },
        "sigmoid": {
            "pr_auc": round(pr_auc(test_labels, sigmoid_probabilities), 6),
            "brier": round(sigmoid_report.brier, 6),
            "expected_calibration_error": round(sigmoid_report.expected_calibration_error, 6),
            "distinct_scores": int(np.unique(sigmoid_probabilities).size),
        },
    }


def plot_calibration(report: CalibrationReport, path: Path) -> None:
    """Write the calibration curve. Agg backend only -- nothing here opens a window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 6))
    predicted = [row["mean_predicted"] for row in report.bins]
    observed = [row["observed_frequency"] for row in report.bins]
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    axis.plot(predicted, observed, marker="o", label="meta-learner")
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("observed fraud frequency")
    axis.set_title(
        f"Meta-learner calibration (held-out test)\n"
        f"Brier {report.brier:.4f}, ECE {report.expected_calibration_error:.4f}"
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info("wrote %s", path)


def run(
    settings: Settings,
    *,
    sample: int | None = None,
    skip_registry: bool = False,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_path: Path | None = None,
    blocks_override: Sequence[str] | None = None,
) -> MetaRunReport:
    """Build the features, run the ablation, fit the shipped model and score test once."""
    logger.info("random seed = %d", RANDOM_SEED)
    frames = load_splits(settings.processed_data_dir, SOURCE, sample)

    registered = load_registered_tier1(artifact_dir, registry_path)
    if registered is None:
        raise RuntimeError(
            "the registered Tier-1 model could not be loaded from "
            f"{artifact_dir}. Phase 5 fuses Tier-1's output and cannot run without it."
        )
    logger.info("tier-1: %s (best_iteration=%d)", registered.model_id, registered.best_iteration)

    oof = build_oof_tier1(frames["train"], frames["val"], source=SOURCE, registered=registered)

    tier2: Tier2Features | None = build_tier2_errors(
        frames, artifact_dir=artifact_dir, model_id=TIER2_MODEL_ID
    )

    tier3_sets: list[Tier3Features] = []
    circularity: dict[str, Any] | None = None
    corpus = (
        pd.concat([frame.assign(split=split) for split, frame in frames.items()], ignore_index=True)
        .sort_values("event_time", kind="stable")
        .reset_index(drop=True)
    )
    try:
        from app.models.tier3_edges import IEEE_FINGERPRINTS, NON_CIRCULAR_FINGERPRINTS
        from app.models.train_tier3 import (
            IEEE_CADENCE,
            IEEE_WINDOW,
            ieee_factory,
            split_boundaries,
        )

        boundaries = split_boundaries(corpus)
        main = build_tier3_features(
            corpus,
            factory=ieee_factory(200, IEEE_FINGERPRINTS),
            boundaries=boundaries,
            cadence=IEEE_CADENCE,
            window=IEEE_WINDOW,
            prefix="tier3_",
        )
        tier3_sets.append(main)
        control = build_tier3_features(
            corpus,
            factory=ieee_factory(200, NON_CIRCULAR_FINGERPRINTS),
            boundaries=boundaries,
            cadence=IEEE_CADENCE,
            window=IEEE_WINDOW,
            prefix="tier3nc_",
        )
        tier3_sets.append(control)
        circularity = {
            "note": (
                "The control uses fingerprints sharing no column with the constructed "
                "account_id, at a MATCHED entity cap of 200. Phase 4 defect B5 recorded its "
                "own control as confounded by a mismatched cap (main 200 against control 50); "
                "matching it here closes that defect rather than inheriting it."
            ),
            "main_abstention_rate": round(main.abstention_rate, 6),
            "control_abstention_rate": round(control.abstention_rate, 6),
        }
    except Exception as exc:  # noqa: BLE001 - a stated gap, never a silent skip
        logger.error("tier-3 feature construction failed: %s", exc)
        logger.error("the ablation will run without any tier3 block, and the report says so")

    frame = assemble_meta_frame(
        frames, oof=oof, registered=registered, tier2=tier2, tier3=tier3_sets
    )

    tier2_diagnostic: dict[str, Any] | None = None
    tier2_reading: list[str] = []
    if tier2 is not None:
        train_rows = frame[frame["split"] == "train"]
        measured = train_rows.set_index("transaction_id")["tier2_error"].where(
            train_rows.set_index("transaction_id")["tier2_is_scoreable"].astype(bool)
        )
        diagnostic = tier2_memorisation_diagnostic(frames["train"], measured)
        tier2_diagnostic = diagnostic.to_dict()
        tier2_reading = diagnostic.describe()
        for line in tier2_reading:
            logger.info("tier2 diagnostic: %s", line)

    # Block 1 has no honest out-of-fold Tier-1 score. Dropped, not filled.
    train_fit = frame[(frame["split"] == "train") & frame[OOF_SCOREABLE_COLUMN]].copy()
    validation = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()
    slices = validation_slices(validation)

    blocks = list(blocks_override or available_blocks(frame))
    logger.info("feature blocks available: %s", blocks)

    ablation_validation = run_ablation(slices.fit, slices.arbiter, blocks, label="validation-fit")
    ablation_train = run_ablation(train_fit, slices.arbiter, blocks, label="train-fit(OOF)")
    retained = list(ablation_validation.retained)
    logger.info("retained after arbitration: %s", retained)

    # --- the shipped model ---------------------------------------------------------------
    matrix, names = matrix_for(train_fit, retained, "tier3_")
    validation_matrix, _ = matrix_for(slices.fit, retained, "tier3_")
    booster = fit_booster(
        matrix,
        _labels(train_fit),
        feature_names=names,
        rounds=XGBOOST_NUM_ROUNDS,
        validation=(validation_matrix, _labels(slices.fit)),
        early_stopping=XGBOOST_EARLY_STOPPING,
    )

    # Early stopping chose this on V-fit; scoring must honour it or the selection was pointless.
    best_iteration = getattr(booster, "best_iteration", None)
    iteration_range = (0, 0) if best_iteration is None else (0, int(best_iteration) + 1)
    logger.info(
        "shipped booster: early stopping selected iteration %s; scoring with range %s",
        best_iteration,
        iteration_range,
    )
    late_probabilities = crossfit_calibrated_scores(booster, slices.late, names, iteration_range)
    median_amount = float(np.median(_amounts(frames["train"])))
    cost_model = CostModel.scaled_to(median_amount, IEEE_COST_UNITS)
    operating = choose_operating_points(
        _labels(slices.late), late_probabilities, _amounts(slices.late), cost_model
    )
    calibrator = fit_calibrator(
        booster,
        slices.late.loc[:, list(names)].to_numpy(dtype="float64"),
        _labels(slices.late),
        feature_names=names,
        iteration_range=iteration_range,
    )

    spec = build_spec(
        retained,
        # Only the layers the shipped model actually reads. Recording a tier whose block the
        # ablation retired overstates what this model consumes, and a reader tracing a decision
        # back through them would find inputs that never reached the matrix.
        tier_model_versions={
            "tier1": registered.model_id,
            **({"tier2": tier2.model_id} if tier2 is not None and "tier2" in retained else {}),
            **(
                {"tier3": "refit-from-snapshots"}
                if tier3_sets and any(b.startswith("tier3") for b in retained)
                else {}
            ),
        },
        upstream_feature_versions={
            "pipeline": str(frames["train"]["feature_version"].iloc[0]),
            "tier1": registered.model.spec.to_feature_definition().feature_version,
            **({"tier2": tier2.feature_version} if tier2 is not None else {}),
        },
    )
    model = MetaModel(
        model_id=build_model_id("meta-learner", "xgboost", SOURCE),
        spec=spec,
        booster=booster,
        calibrator=calibrator,
        review_threshold=operating.review_threshold,
        block_threshold=operating.block_threshold,
        best_iteration=None if best_iteration is None else int(best_iteration),
        hyperparameters={
            **XGBOOST_PARAMS,
            "num_boost_round": XGBOOST_NUM_ROUNDS,
            "early_stopping_rounds": XGBOOST_EARLY_STOPPING,
            "best_iteration": (
                XGBOOST_NUM_ROUNDS if best_iteration is None else int(best_iteration)
            ),
            "trees_scored_with": iteration_range[1] or "all",
            "calibration": "sigmoid (Platt)",
            "calibration_rows": len(slices.late),
            "calibration_positives": int(_labels(slices.late).sum()),
            "oof_scheme": "expanding-window forward chaining",
            "oof_blocks": OOF_BLOCKS,
            "oof_dropped_block": 1,
            "oof_tier1_rounds": oof.rounds,
            "oof_rank_normalised": True,
            "ablation_rounds": ABLATION_NUM_ROUNDS,
            "ablation_max_depth": ABLATION_MAX_DEPTH,
            "review_threshold": operating.review_threshold,
            "block_threshold": operating.block_threshold,
        },
    )

    # --- test, scored exactly once --------------------------------------------------------
    test_labels = _labels(test)
    test_probabilities = model.score_frame(test)
    test_cost = cost_at_threshold(
        test_labels,
        test_probabilities,
        _amounts(test),
        operating.review_threshold,
        cost_model,
    )
    result = evaluate(
        "meta-learner (XGBoost fusion, Platt-scaled)",
        "test",
        test_labels,
        test_probabilities,
        threshold=operating.review_threshold,
        threshold_criterion=operating.review_criterion,
        interval=bootstrap_pr_auc(test_labels, test_probabilities, seed=RANDOM_SEED),
        cost=test_cost,
        notes=[
            f"Retained blocks: {', '.join(retained)}.",
            "Every selection was made on validation: the retained feature blocks on V-arb, "
            "the calibrator and both thresholds on V-late, the early-stopping round on V-fit. "
            "Test selected nothing. It was, however, read more than once and the honest "
            "phrasing is not 'scored exactly once': the isotonic baseline is a second scoring, "
            "the matched-flag-rate table is a third, and the run was repeated after a "
            "score-quantisation bug was found. That repeat moved the meta-learner by 0.00002 "
            "and corrected the baseline by 0.0074, i.e. it made the shipped result less "
            "flattering, not more.",
        ],
    )

    # The registered model's own scores, not the CDF-mapped column the meta-learner reads. The
    # map exists to make the meta-learner's *input* commensurable with the out-of-fold ranks it
    # was fitted on; the head-to-head baseline is what Tier-1 would actually serve, and it
    # reproduces Phase 2's published 0.5276 exactly.
    tier1_test = np.asarray(registered.model.score_frame(frames["test"]), dtype="float64")
    # The baseline's operating point is chosen on VALIDATION, exactly like the meta-learner's.
    # Taking a quantile of the test scores instead -- which an earlier version of this function
    # did -- hands Tier-1 the test flag rate while the meta-learner has to transfer its threshold
    # blind from V-late. Precision falls with flag rate, so that comparison flatters whichever
    # model was allowed to peek, and every threshold-dependent figure it produces is unreportable.
    tier1_validation = np.asarray(registered.model.score_frame(frames["val"]), dtype="float64")
    tier1_threshold = threshold_for_flag_rate(tier1_validation, MAX_REVIEW_FLAG_RATE)
    tier1_result = evaluate(
        f"Tier-1 alone ({registered.model_id})",
        "test",
        test_labels,
        tier1_test,
        threshold=tier1_threshold,
        threshold_criterion=(
            f"flagging at most {MAX_REVIEW_FLAG_RATE:.1%} of VALIDATION traffic, transferred to "
            "test unchanged -- the same discipline the meta-learner threshold is held to"
        ),
        interval=bootstrap_pr_auc(test_labels, tier1_test, seed=RANDOM_SEED),
        cost=cost_at_threshold(
            test_labels, tier1_test, _amounts(test), tier1_threshold, cost_model
        ),
    )

    # Both models at a matched flag rate, so the precision comparison means something. Without
    # this the two are read side by side while flagging different shares of traffic.
    matched = {}
    for label, scores in (("meta", test_probabilities), ("tier1", tier1_test)):
        cut = threshold_for_flag_rate(scores, MAX_REVIEW_FLAG_RATE)
        estimate = cost_at_threshold(test_labels, scores, _amounts(test), cut, cost_model)
        confusion = confusion_at_threshold(test_labels, scores, cut)
        # Value recall, not just count recall. It is the mechanism behind the cost difference --
        # a model can miss more frauds and still cost less by missing cheaper ones -- so it has
        # to be produced by the run rather than reconstructed by hand afterwards.
        flagged = scores >= cut
        amounts = _amounts(test)
        fraud_value = float(amounts[test_labels].sum())
        caught_value = float(amounts[test_labels & flagged].sum())
        matched[label] = {
            "threshold": cut,
            "flag_rate": round(estimate.flag_rate, 6),
            "precision": round(confusion.precision, 6),
            "recall": round(confusion.recall, 6),
            "recall_by_value": round(caught_value / fraud_value if fraud_value else 0.0, 6),
            "caught_fraud_value": round(caught_value, 2),
            "missed_fraud_value": round(fraud_value - caught_value, 2),
            "confusion_matrix": confusion.to_dict(),
            "cost_per_1000": round(estimate.cost_per_1000_units, 2),
            "note": (
                "Matched-flag-rate view. BOTH cuts here are quantiles of the TEST score vectors "
                "-- that is what makes the flag rates comparable, and it is why this table "
                "selects nothing and is reported only after the shipped thresholds were fixed "
                "on validation. The shipped operating points are elsewhere in this entry."
            ),
        }
    logger.info(
        "matched 1%% flag rate -- meta precision %.4f / value-recall %.4f / cost %.2f vs "
        "tier1 precision %.4f / value-recall %.4f / cost %.2f per 1,000",
        matched["meta"]["precision"],
        matched["meta"]["recall_by_value"],
        matched["meta"]["cost_per_1000"],
        matched["tier1"]["precision"],
        matched["tier1"]["recall_by_value"],
        matched["tier1"]["cost_per_1000"],
    )
    delta_interval = bootstrap_pr_auc_delta(
        test_labels, test_probabilities, tier1_test, seed=RANDOM_SEED
    )
    delta_point = result.pr_auc - tier1_result.pr_auc

    calibration = calibration_curve_points(test_labels, test_probabilities)
    isotonic = isotonic_comparison(
        booster, slices.late, names, test, test_probabilities, iteration_range
    )

    report = MetaRunReport(
        model=model,
        result=result,
        tier1_result=tier1_result,
        delta_vs_tier1=delta_interval,
        delta_point=delta_point,
        ablation_validation=ablation_validation,
        ablation_train=ablation_train,
        calibration=calibration,
        isotonic_comparison=isotonic,
        operating=operating,
        slices=slices.describe(),
        handicap=[
            {
                "fold": row.fold,
                "train_rows": row.train_rows,
                "train_positives": row.train_positives,
                "validation_pr_auc": round(row.validation_pr_auc, 6),
            }
            for row in oof.handicap
        ]
        + [
            {
                "fold": "full",
                "train_rows": len(frames["train"]),
                "train_positives": int(_labels(frames["train"]).sum()),
                "validation_pr_auc": round(oof.full_train_validation_pr_auc, 6),
            }
        ],
        tier2_diagnostic=tier2_diagnostic,
        tier2_diagnostic_reading=tier2_reading,
        best_iteration=("all" if best_iteration is None else int(best_iteration)),
        tier3_circularity=circularity,
        latency={},
        matched_flag_rate=matched,
        training_window={
            "start": str(train_fit["event_time"].min()),
            "end": str(train_fit["event_time"].max()),
        },
        notes=[],
    )

    logger.info("\n%s", result.render())
    logger.info("\n%s", tier1_result.render())
    logger.info(
        "meta minus Tier-1 alone: %+.4f, 95%% CI [%+.4f, %+.4f] -- %s",
        delta_point,
        delta_interval[0],
        delta_interval[1],
        "a tie" if delta_interval[0] <= 0.0 <= delta_interval[1] else "excludes zero",
    )

    reports_dir = report_path.parent if report_path else settings.reports_dir
    plot_calibration(calibration, reports_dir / "meta_calibration_curve.png")
    written = report_path or (settings.reports_dir / "meta_report.md")
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(render_report(report, sample=sample), encoding="utf-8")
    logger.info("wrote %s", written)

    if not skip_registry:
        register(report, oof, registry_path, artifact_dir)
    return report


def register(
    report: MetaRunReport,
    oof: Any,
    registry_path: Path,
    artifact_dir: Path,
) -> RegistryEntry:
    """Save the model and append its permanent record."""
    artifact = report.model.save(artifact_dir)
    # Read the ids rather than hardcoding them: a literal list goes stale on the next run and
    # then every future entry names a two-generations-old model as the thing it supersedes.
    # Describe each superseded entry from what it actually contains rather than from a literal
    # that goes stale the moment another defect is found. The previous version listed three
    # defects by hand and silently omitted the calibration bug that produced the very entry it
    # was superseding -- and that entry's cost figure is the more flattering one.
    superseded_notes: list[str] = []
    for past in read_registry(registry_path):
        if past.get("layer") != "meta_learner":
            continue
        heldout = past.get("heldout_test", {}) or {}
        calibration = heldout.get("calibration", {}) or {}
        baselines = past.get("baseline_comparison") or [{}]
        defects: list[str] = []
        threshold = float(heldout.get("threshold", 0.0) or 0.0)
        if 0.0 < threshold < 1e-6:
            defects.append(
                "a calibrator fitted on the probability scale and applied to margins, so every "
                "probability and threshold in it is an artefact"
            )
        if float(calibration.get("expected_calibration_error", 0.0) or 0.0) > 0.02:
            defects.append("uninterpretable calibration")
        if "of test traffic" in str(baselines[0].get("threshold_criterion", "")):
            defects.append("a Tier-1 baseline thresholded on the test split")
        if abs(float(baselines[0].get("pr_auc", 0.5276) or 0.5276) - 0.5276) > 1e-4:
            defects.append("a quantisation-damaged Tier-1 baseline")
        for verdict in (heldout.get("ablation_validation", {}) or {}).get("verdicts", []):
            rationale = str(verdict.get("rationale", ""))
            if "does not exclude zero" in rationale and "-0.0" in rationale:
                defects.append(
                    "a retirement sentence claiming an interval does not exclude zero when it "
                    "excludes zero negatively"
                )
                break
        superseded_notes.append(
            f"{past['model_id']} ({', '.join(defects) if defects else 'superseded'})"
        )
    caveats: list[str] = []
    if report.result.is_leak_suspicious:
        caveats.append(
            f"DO NOT QUOTE AS A HEADLINE. Test PR-AUC {report.result.pr_auc:.4f} exceeds the "
            "leak-suspicion floor on fraud data, which ml-evaluation-standards section 4 treats "
            "as a leak signal until disproven. The meta-learner is the likeliest place in this "
            "project to trip it, because it fuses four signal sources. Investigate before use."
        )

    delta = report.delta_vs_tier1
    tie = delta[0] <= 0.0 <= delta[1]
    notes = [
        *caveats,
        (
            (
                "Supersedes every earlier meta_learner entry: "
                + "; ".join(superseded_notes)
                + ". The registry is append-only, so those entries remain, carrying the "
                "defects they were produced with and no forward pointer of their own. "
                "Supersession can only be recorded forward, and this is that record. Treat "
                "any threshold-dependent figure in them as an artefact of the defect named "
                "beside it."
            )
            if superseded_notes
            else "First meta_learner entry."
        ),
        (
            f"Headline against Tier-1 alone: {report.delta_point:+.4f}, 95% CI "
            f"[{delta[0]:+.4f}, {delta[1]:+.4f}] -- "
            + (
                "the interval includes zero, so the meta-learner TIES Tier-1 on this test split."
                if tie
                else "the interval excludes zero."
            )
        ),
        (
            f"Out-of-fold Tier-1: {OOF_BLOCKS} forward-chaining blocks over the train split, "
            f"block 1 dropped (no predecessor to score it), {oof.rounds} rounds per fold fixed "
            "from the registered model's best_iteration, each fold refitting its own input "
            f"spec. Fold feature versions: {list(oof.feature_versions)}."
        ),
        *[verdict.rationale() for verdict in report.ablation_validation.verdicts],
        (
            "Keep/drop was arbitrated on the V-arb validation slice, where no tier signal is "
            "in-sample. The train-fit column is reported alongside; the disagreement between "
            "the two is the measured size of Tier-2's train contamination."
        ),
        (
            "Tier-2's latent size, window and early-stopping round were all selected on "
            "validation, so validation flatters Tier-2 relative to test. A drop verdict is "
            "therefore safe (it lost with a thumb on the scale); a keep verdict is suggestive "
            "rather than conclusive."
        ),
        (
            "Tier-1's round count was early-stopped on the full validation split during Phase 2, "
            "so all three validation slices carry one Tier-1 hyperparameter tuned on them. It "
            "is one scalar over 88,581 rows and cannot be undone without retraining Tier-1."
        ),
        (
            "tier3_topology is not obtainable from a Tier3Result today. If it was retained "
            "above, serving it is a Phase 7 requirement: Tier-3's persisted score table must "
            "carry a per-account feature vector."
        ),
        (
            "Not fitted on PaySim. Its Tier-1 PR-AUC of 0.9999 is a simulator artefact "
            "(amount == oldbalanceOrg on 97.49% of fraud), it has no Tier-2 model, and Tier-3 "
            "abstains on 100% of its test transactions. See BUILD_LOG.md Phase 2 finding 3."
        ),
        (
            "top_features is an evasion oracle: authenticated internal reviewers only, never "
            "returned to the transacting party."
        ),
    ]
    if report.tier2_diagnostic is not None:
        notes.append(f"Tier-2 memorisation diagnostic: {report.tier2_diagnostic}.")
    if report.tier3_circularity is not None:
        notes.append(f"Tier-3 circularity control: {report.tier3_circularity['note']}")

    entry = RegistryEntry(
        model_id=report.model.model_id,
        layer="meta_learner",
        algorithm=report.model.algorithm,
        source_dataset=SOURCE,
        feature_version=report.model.spec.feature_version(),
        training_window=report.training_window,
        hyperparameters=report.model.hyperparameters,
        random_seed=RANDOM_SEED,
        heldout_test={
            **report.result.to_dict(),
            "calibration": report.calibration.to_dict(),
            "calibration_baseline_isotonic": report.isotonic_comparison,
            "delta_vs_tier1_alone": {
                "delta": round(report.delta_point, 6),
                "interval": [
                    round(report.delta_vs_tier1[0], 6),
                    round(report.delta_vs_tier1[1], 6),
                ],
                "verdict": "tie" if tie else "excludes zero",
            },
            "matched_flag_rate_comparison": report.matched_flag_rate,
            "ablation_validation": report.ablation_validation.to_dict(),
            "ablation_train_oof": (
                report.ablation_train.to_dict() if report.ablation_train else None
            ),
            "out_of_fold_handicap": report.handicap,
            "validation_slices": report.slices,
            "tier2_memorisation": report.tier2_diagnostic,
            "tier3_circularity": report.tier3_circularity,
            "latency": report.latency,
        },
        baseline_comparison=[report.tier1_result.to_dict()],
        artifact=str(artifact.relative_to(artifact_dir.parent)),
        notes=notes,
    )
    append_entry(entry, registry_path)
    logger.info("appended %s to %s", entry.model_id, registry_path)
    return entry


def render_report(report: MetaRunReport, sample: int | None = None) -> str:
    """Render the full metrics report."""
    flagged_rows = report.result.confusion.false_positives + report.result.confusion.true_positives
    realised_flag_rate = (
        flagged_rows / report.result.confusion.total if report.result.confusion.total else 0.0
    )
    lines: list[str] = ["# Phase 5 — Meta-Learner (XGBoost fusion)", ""]
    if sample is not None:
        lines += [
            f"> **Sampled run**: only the earliest {sample:,} rows of each split were used. "
            "These numbers are for iteration and are not reportable.",
            "",
        ]
    lines += [
        f"Random seed {RANDOM_SEED}. Model `{report.model.model_id}`.",
        "",
        "## Held-out test",
        "",
        "```",
        report.result.render(),
        "```",
        "",
        "### Tier-1 alone, same test split",
        "",
        "```",
        report.tier1_result.render(),
        "```",
        "",
        f"**Meta-learner minus Tier-1 alone: {report.delta_point:+.4f}, 95% CI "
        f"[{report.delta_vs_tier1[0]:+.4f}, {report.delta_vs_tier1[1]:+.4f}].**",
        "",
    ]
    if report.delta_vs_tier1[0] <= 0.0 <= report.delta_vs_tier1[1]:
        lines += [
            "The interval includes zero. **The meta-learner ties Tier-1 on this test split.** "
            "That is the honest reading, and it is the outcome Phases 3 and 4 predicted: "
            "Tier-1 carries the system and the other layers add little. It is a finding, not "
            "a failure.",
            "",
        ]

    lines += [
        "### Both models at a matched 1% flag rate",
        "",
        "The two operating points above flag different shares of traffic, and precision falls as "
        "flag rate rises, so reading their precisions side by side compares two different "
        "things. This table holds the flag rate fixed. **Both cuts are quantiles of the test "
        "score vectors**, which is what makes the flag rates comparable. It is computed "
        "after the shipped thresholds were already fixed on validation, and it selects "
        "nothing.",
        "",
        "| model | flag rate | precision | recall (count) | recall (value) | cost per 1,000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("meta", "tier1"):
        row = report.matched_flag_rate[name]
        lines.append(
            f"| {name} | {row['flag_rate']:.2%} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['recall_by_value']:.4f} | "
            f"{row['cost_per_1000']:,.2f} |"
        )
    lines += ["", "## The ablation — which layers paid for themselves", "", "### Verdicts", ""]
    for verdict in report.ablation_validation.verdicts:
        lines.append(f"- {verdict.rationale()}")
    lines += [
        "",
        "### Paired deltas, measured on the V-arb validation slice",
        "",
        "| block | direction | delta | 95% CI | width | verdict |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for delta in report.ablation_validation.deltas:
        lines.append(
            f"| {delta.block} | {delta.kind} | {delta.delta:+.4f} | "
            f"[{delta.interval[0]:+.4f}, {delta.interval[1]:+.4f}] | {delta.width:.4f} | "
            f"{'keeps' if delta.excludes_zero_positively else 'no evidence'} |"
        )
    lines += [
        "",
        f"Arbiter slice: {report.ablation_validation.arbiter_rows:,} rows, "
        f"{report.ablation_validation.arbiter_positives:,} positives. Interval widths above are "
        "the honest read on what this comparison could have detected. A retirement here is an "
        "absence of evidence at this sample size, not evidence the layer is worthless.",
        "",
    ]
    if report.ablation_train is not None:
        lines += [
            "### The same leave-one-out deltas, fitted on the train (out-of-fold) split",
            "",
            "Reported so the two columns can be compared. Tier-2 is contaminated on train by an "
            "account-level, label-selected eligibility filter, so a disagreement between this "
            "table and the one above is a measurement of that contamination.",
            "",
            "| block | direction | delta | 95% CI |",
            "| --- | --- | ---: | --- |",
        ]
        for delta in report.ablation_train.deltas:
            lines.append(
                f"| {delta.block} | {delta.kind} | {delta.delta:+.4f} | "
                f"[{delta.interval[0]:+.4f}, {delta.interval[1]:+.4f}] |"
            )
        lines.append("")

    if report.tier2_diagnostic is not None:
        lines += [
            "## Tier-2 memorisation diagnostic",
            "",
            "The autoencoder was fitted on fraud-free train windows, excluded **by account**. "
            "Clean rows belonging to fraud-touching accounts were therefore withheld from the "
            "fit for a reason unrelated to their own label, which makes them a control group "
            "that holds the label constant and varies only fit membership.",
            "",
            "| quantity | value |",
            "| --- | ---: |",
        ]
        for key, value in report.tier2_diagnostic.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines += [f"- {line}" for line in report.tier2_diagnostic_reading]
        residual = float(report.tier2_diagnostic.get("residual_auc", 0.5))
        if residual < 0.5:
            lines += [
                "",
                f"**Read the residual figure carefully: {residual:.4f} is below 0.5.** With fit "
                "membership held constant, Tier-2 scores fraud as *less* anomalous than clean "
                "traffic. Its apparent train-split discrimination is not weak signal - it is the "
                "autoencoder recognising rows it was fitted on, and underneath that the signal "
                "points the wrong way.",
            ]
        lines.append("")

    lines += [
        "## Calibration",
        "",
        f"Brier {report.calibration.brier:.4f}, expected calibration error "
        f"{report.calibration.expected_calibration_error:.4f}. Curve: "
        "`meta_calibration_curve.png`.",
        "",
        "| bin | count | mean predicted | observed frequency |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report.calibration.bins:
        lines.append(
            f"| [{row['bin_lower']:.1f}, {row['bin_upper']:.1f}) | {int(row['count'])} | "
            f"{row['mean_predicted']:.4f} | {row['observed_frequency']:.4f} |"
        )
    lines += [
        "",
        "### Isotonic, the alternative that was not shipped",
        "",
        report.isotonic_comparison["verdict"],
        "",
        f"- isotonic: {report.isotonic_comparison['isotonic']}",
        f"- sigmoid: {report.isotonic_comparison['sigmoid']}",
        "",
        "## Operating points",
        "",
        f"- Review at `{report.operating.review_threshold:.6e}` — "
        f"{report.operating.review_criterion}.",
        f"- Block at `{report.operating.block_threshold:.6e}` — "
        f"{report.operating.block_criterion}.",
        "",
        "Thresholds are printed in scientific notation because an earlier build of this layer "
        "emitted probabilities many orders of magnitude below 1, which fixed-point formatting "
        "rendered as `0.000000`. That was a calibrator fitted on one scale and applied to "
        "another, not a property of the data; it is fixed, and the notation is kept only so a "
        "recurrence stays visible.",
        "",
        "```",
        report.operating.cost.render(),
        "```",
        "",
        "```",
        report.operating.sensitivity,
        "```",
        "",
        "```",
        report.operating.review_sweep,
        "```",
        "",
        "## Measurement design",
        "",
        "### Validation, cut three ways",
        "",
        "| slice | rows | positives | base rate | first event | last event |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report.slices:
        lines.append(
            f"| {row['slice']} | {row['rows']:,} | {row['positives']:,} | "
            f"{row['base_rate']:.4%} | {row['first_event']} | {row['last_event']} |"
        )
    lines += [
        "",
        "### The out-of-fold handicap",
        "",
        "Fold models are fitted on less data than the model that serves, so the out-of-fold "
        "Tier-1 column is noisier than the column the meta-learner meets at test. Measured on "
        "the untouched validation split rather than argued away:",
        "",
        "| fold | train rows | train positives | validation PR-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report.handicap:
        lines.append(
            f"| {row['fold']} | {row['train_rows']:,} | {row['train_positives']:,} | "
            f"{row['validation_pr_auc']:.4f} |"
        )
    lines += [
        "",
        "The bias this introduces is conservative in direction: a noisier Tier-1 column makes "
        "the meta-learner rely on it *less* than it should, so the scheme should cost PR-AUC "
        "rather than manufacture it. That is an argument, not a measurement - it is not "
        "independently verified here, and it is load-bearing in the conclusion.",
        "",
        "## Limitations",
        "",
        f"- **The shipped booster early-stopped at iteration {report.best_iteration}**, and "
        "scoring uses exactly that many trees. A model whose validation metric peaks that early "
        "has very little to learn beyond its strongest input, which is consistent with the "
        "ablation - but it means the loss against Tier-1 is **confounded** between the "
        "out-of-fold handicap and simply not fitting. Separating them needs a diagnostic run "
        "fitted on the in-sample Tier-1 column, which was not performed.",
        "- **The two ablation columns differ in fit size as well as in contamination** (330,703 "
        "rows against 35,432), so their disagreement is not cleanly attributable to either.",
        "- **Tier-3's ring scorer is in-sample on train**, with no out-of-fold remedy. It does "
        "not reach the shipped model, which retains no Tier-3 column, but it is the likely "
        "reason the train-fitted column rates Tier-3 far higher than the arbiter does.",
        "- **The aggregate calibration figure flatters the band the threshold sits in.** "
        f"Expected calibration error is {report.calibration.expected_calibration_error:.4f} "
        f"against a base rate of {report.result.base_rate:.4%}, but that average is dominated by "
        "the near-zero bin holding the overwhelming majority of rows. In the sparse "
        "high-probability bands where the operating threshold actually falls, predicted and "
        "observed diverge materially -- read the per-bin table above, not the summary.",
        f"- **The shipped threshold overshoots the capacity cap it is named after.** It is "
        f"described as respecting a {MAX_REVIEW_FLAG_RATE:.0%} review cap, chosen on "
        f"V-late, but realises {realised_flag_rate:.2%} on test. The cap binds on the split "
        "it was chosen on; transferring a threshold across a base-rate shift does not "
        "preserve the flag rate, and nothing re-checks it downstream.",
        "- **Latency was not benchmarked**; the registry entry carries an empty latency block.",
        "- **No false-negative profiling code exists for this layer**, unlike Tiers 2 and 3. The "
        "observed-failure analysis in `app/models/README.md` was computed by hand and does not "
        "regenerate with this report.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 5 training from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m app.models.train_meta_learner",
        description="Train, ablate and register the RiskIQ meta-learner.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override DATA_DIR.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Earliest N rows per split. For iteration only; results are not reportable, and "
        "Tier-2 coverage and Tier-3 snapshots both degrade because histories are truncated.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Do not save artefacts or append to models/registry.json.",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=None,
        help="Force a feature-block set, bypassing availability detection. For debugging.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the metrics report (default: notebooks/meta_report.md).",
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

    try:
        report = run(
            settings,
            sample=args.sample,
            skip_registry=args.skip_registry,
            report_path=args.report or (settings.reports_dir / "meta_report.md"),
            blocks_override=args.blocks,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("meta-learner training failed: %s", exc)
        return 1

    if report.result.is_leak_suspicious:
        logger.error(
            "test PR-AUC %.4f is leak-suspicious; the registry entry carries the caveat",
            report.result.pr_auc,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
