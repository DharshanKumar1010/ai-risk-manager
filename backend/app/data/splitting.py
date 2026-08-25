"""Chronological train/val/test assignment.

Never a shuffle. Fraud data is time-ordered and IEEE-CIS propagates a reported chargeback
across an account's *subsequent* transactions, so a random split trains on the future and
scores itself on the past — the classic way to produce a PR-AUC that looks excellent and
means nothing.

Two details that are easy to get wrong:

**Ties at the boundary.** Splitting purely by row position puts identical timestamps on
both sides of a boundary, which is an overlap however you label it. PaySim is at one-hour
granularity across 6.36M rows, so boundary timestamps are shared by thousands of rows. The
cut is therefore chosen as a *timestamp*, and membership is decided by strict comparison
against it. Split sizes then deviate slightly from 70/15/15, which is the honest outcome —
:meth:`SplitBoundaries.fractions` reports what was actually produced.

**Per source.** IEEE-CIS and PaySim clocks share no origin, so a boundary is only meaningful
within one corpus. Each is split independently against its own timeline.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.data.raw_spec import SourceDataset
from app.data.schema import SPLIT_FRACTIONS, SPLIT_ORDER, Split

logger = logging.getLogger("riskiq.splitting")


@dataclass(frozen=True)
class SplitBoundaries:
    """Where one corpus's timeline was cut, and what that produced.

    Attributes:
        source_dataset: The corpus these boundaries describe.
        train_end: First timestamp *not* in train. Train is ``event_time < train_end``.
        val_end: First timestamp *not* in val.
        counts: Rows landing in each split.
    """

    source_dataset: SourceDataset
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    counts: Mapping[Split, int]

    @property
    def total(self) -> int:
        """Return the total row count across all splits."""
        return sum(self.counts.values())

    @property
    def fractions(self) -> dict[Split, float]:
        """Return the fraction of rows actually landing in each split."""
        total = self.total
        return {split: (self.counts[split] / total if total else 0.0) for split in SPLIT_ORDER}


def assign_splits(
    event_time: pd.Series,
    source_dataset: SourceDataset,
    fractions: Mapping[Split, float] = SPLIT_FRACTIONS,
) -> tuple[pd.Series, SplitBoundaries]:
    """Assign each row to a split by time, with strictly disjoint boundaries.

    Args:
        event_time: Timezone-aware timestamps, in any row order.
        source_dataset: The corpus being split, recorded on the boundaries.
        fractions: Target proportions. Only ``train`` and ``val`` are read; ``test`` takes
            the remainder, so the splits always cover every row exactly once.

    Returns:
        A ``(split, boundaries)`` pair. ``split`` is aligned to ``event_time``'s index.

    Raises:
        ValueError: If there are too few rows to split, or if boundary ties would leave a
            split empty — a silently empty validation split would invalidate every threshold
            chosen on it.
    """
    if event_time.isna().any():
        raise ValueError("event_time contains nulls; a row with no timestamp cannot be split")

    row_count = len(event_time)
    if row_count < len(SPLIT_ORDER):
        raise ValueError(f"Need at least {len(SPLIT_ORDER)} rows to split, got {row_count}")

    ordered = np.sort(event_time.to_numpy())
    train_cut = min(int(row_count * fractions["train"]), row_count - 1)
    val_cut = min(int(row_count * (fractions["train"] + fractions["val"])), row_count - 1)

    train_end = pd.Timestamp(ordered[train_cut])
    val_end = pd.Timestamp(ordered[val_cut])

    if train_end == val_end:
        raise ValueError(
            f"{source_dataset}: the train and validation boundaries fall on the same "
            f"timestamp ({train_end}), which would leave the validation split empty. The "
            "timeline is too coarse relative to its row count to cut at these fractions."
        )

    split = pd.Series(
        np.where(
            event_time < train_end,
            "train",
            np.where(event_time < val_end, "val", "test"),
        ),
        index=event_time.index,
        dtype="object",
    )

    counts: dict[Split, int] = {name: int((split == name).sum()) for name in SPLIT_ORDER}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"{source_dataset}: split(s) {empty} are empty after assignment")

    return split, SplitBoundaries(
        source_dataset=source_dataset,
        train_end=train_end,
        val_end=val_end,
        counts=counts,
    )


@dataclass(frozen=True)
class SplitWindow:
    """The observed time range and class balance of one split."""

    split: Split
    rows: int
    first_event: pd.Timestamp
    last_event: pd.Timestamp
    positives: int

    @property
    def base_rate(self) -> float:
        """Return the positive-class rate. Quoted alongside every metric downstream."""
        return self.positives / self.rows if self.rows else 0.0


def summarise_splits(frame: pd.DataFrame) -> list[SplitWindow]:
    """Describe each split's observed time range and class balance.

    Used by both the data-quality report and the split-boundary test, so the numbers a
    reader sees are the same ones the assertion checks.

    Args:
        frame: Must carry ``split``, ``event_time`` and ``is_fraud`` columns.
    """
    windows: list[SplitWindow] = []
    for name in SPLIT_ORDER:
        rows = frame.loc[frame["split"] == name]
        if rows.empty:
            continue
        windows.append(
            SplitWindow(
                split=name,
                rows=len(rows),
                first_event=rows["event_time"].min(),
                last_event=rows["event_time"].max(),
                positives=int(rows["is_fraud"].sum()),
            )
        )
    return windows


def find_boundary_overlaps(frame: pd.DataFrame) -> list[str]:
    """Return descriptions of any chronological overlap between consecutive splits.

    Empty means the splits are strictly time-ordered: every train timestamp is earlier than
    every val timestamp, and every val timestamp earlier than every test timestamp.
    """
    windows = {window.split: window for window in summarise_splits(frame)}
    problems: list[str] = []
    # Deliberately uneven: pairs each split with its successor, so the last has no partner.
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        if earlier not in windows or later not in windows:
            continue
        if windows[earlier].last_event >= windows[later].first_event:
            problems.append(
                f"{earlier} ends at {windows[earlier].last_event} but {later} begins at "
                f"{windows[later].first_event} — the boundary is not strict"
            )
    return problems


# ==========================================================================================
# The validation split, cut three ways
# ==========================================================================================
#
# Phase 5 introduced this cut and kept it local to its own driver. Phase 6 needs the same
# three slices for the same reasons -- fit a calibrator somewhere, choose an operating point
# somewhere later, and never on the rows a delta was measured on -- so it lives here, beside
# the train/val/test cut it refines, rather than being imported out of a training script.


#: Chronological cut of the validation split, as fractions of its rows.
#:
#: The arbiter measures a *ranking* difference, which is stable across time, so it takes an
#: early slice. The calibrator and the thresholds are level-sensitive, and the base rate visibly
#: drifts across this corpus, so they take the slice closest to test.
VALIDATION_FRACTIONS: tuple[float, float, float] = (0.40, 0.35, 0.25)


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
