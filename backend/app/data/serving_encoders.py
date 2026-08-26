"""The Phase 1 frequency-encoder tables, persisted so serving can reproduce them.

**The gap this closes.** ``app.data.pipeline`` fits four frequency encoders on the train split
(``ProductCD``, ``card4``, ``card6``, ``P_emaildomain``), applies them, writes the resulting
``freq_*`` columns into the engineered vector, and then discards the tables — only their digest
survives, folded into the ``feature_version`` hash. That is sufficient for training, which
reads the already-encoded parquet, and insufficient for serving, which is handed a raw
``ProductCD`` and must produce the same ``freq_ProductCD`` the model was fitted against.

Tier-1's *own* encoders (``id_30``, ``id_31``, ``id_33``, ``DeviceInfo``) do not have this
problem: :meth:`Tier1Model.save` writes them into its sidecar, so they load with the model.
These four are the ones that arrive as pre-computed numeric columns from the pipeline, and so
have nowhere to live.

**Why a rebuild rather than a re-run.** The tables are a deterministic function of the train
split — normalised value counts over four columns — so they can be recomputed exactly from the
parquet the pipeline already wrote, without re-running the pipeline or touching the database.
:func:`build_from_processed` does that, and :func:`verify_against_processed` proves it by
re-deriving the ``freq_*`` columns and comparing them to the ones the pipeline itself produced.
A rebuild that cannot reproduce the pipeline's own output is refused rather than shipped.

The artefact is plain JSON. No pickle is loaded on a path reachable from an HTTP endpoint —
the same rule that made Tier-1 persist LightGBM in its native text format.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.data import features as feature_engineering
from app.data.feature_store import digest_encoders
from app.data.raw_spec import SourceDataset
from app.ml.registry import artifact_path

logger = logging.getLogger(__name__)

#: Registry-style identifier for the artefact, per corpus. Not a trained model, so it carries
#: no registry entry; it is a fitted transformer keyed by the feature definition it belongs to.
SERVING_ENCODER_ID = "serving-encoders-{source_dataset}"

#: Tolerance when checking a rebuilt encoding against the pipeline's own output. The stored
#: table is rounded to 12 decimal places by ``digest_encoders``; this is looser than that and
#: far tighter than anything that could change a split in a boosted tree.
ENCODING_TOLERANCE = 1e-9


def encoder_artifact_path(source_dataset: SourceDataset, directory: Path) -> Path:
    """Return the artefact path for one corpus's serving encoders."""
    return artifact_path(
        SERVING_ENCODER_ID.format(source_dataset=source_dataset.replace("_", "-")),
        directory,
        ".json",
    )


def frequency_columns_for(source_dataset: SourceDataset) -> tuple[str, ...]:
    """Return the columns the Phase 1 pipeline frequency-encodes for this corpus."""
    return (
        feature_engineering.IEEE_FREQUENCY_COLUMNS
        if source_dataset == "ieee_cis"
        else feature_engineering.PAYSIM_FREQUENCY_COLUMNS
    )


def build_from_processed(
    train_frame: pd.DataFrame,
    source_dataset: SourceDataset,
) -> dict[str, dict[str, float]]:
    """Refit the pipeline's frequency encoders from the processed train split.

    Args:
        train_frame: The train split as the pipeline wrote it. Only the frequency-encoded
            source columns are read.
        source_dataset: Which corpus's column set to fit.

    Returns:
        Column name to ``{category: frequency}``, the same shape
        :func:`app.data.features.fit_frequency_encoders` returns.
    """
    columns = frequency_columns_for(source_dataset)
    # Every row here is already the train split, so the mask is all-true. Passing the real
    # function rather than reimplementing value counts is the point: if the pipeline's
    # definition of "frequency" ever changes, this changes with it.
    train_mask = pd.Series(True, index=train_frame.index)
    return feature_engineering.fit_frequency_encoders(train_frame, train_mask, columns)


def verify_against_processed(
    encoders: dict[str, dict[str, float]],
    frame: pd.DataFrame,
    source_dataset: SourceDataset,
) -> None:
    """Prove the rebuilt tables reproduce the pipeline's own ``freq_*`` columns.

    The check that makes this artefact trustworthy. It re-encodes the frame and compares each
    generated column against the one the pipeline wrote into the same file. Anything that
    differs means the rebuild is not the thing the model was fitted against, and a model fed a
    subtly different encoding produces a wrong decision underneath a correct-looking audit row.

    Args:
        encoders: The rebuilt tables.
        frame: Any split the pipeline wrote, carrying both the source columns and its own
            ``freq_*`` output.
        source_dataset: Which corpus is being checked.

    Raises:
        ValueError: If any re-derived column disagrees with the pipeline's, or if the frame
            does not carry the columns needed to check.
    """
    encoded = feature_engineering.apply_frequency_encoders(frame.copy(), encoders)
    for column in frequency_columns_for(source_dataset):
        generated = f"freq_{column}"
        if generated not in frame.columns:
            raise ValueError(
                f"{generated} is absent from the processed frame, so the rebuild cannot be "
                "verified against it. Refusing to write an unverified encoder artefact."
            )
        difference = (encoded[generated] - frame[generated]).abs().max()
        if not difference <= ENCODING_TOLERANCE:
            raise ValueError(
                f"rebuilt {generated} differs from the pipeline's by up to {difference:g}, "
                f"above the {ENCODING_TOLERANCE:g} tolerance. The rebuild is not the encoding "
                "the model was fitted against."
            )


def save_serving_encoders(
    encoders: dict[str, dict[str, float]],
    source_dataset: SourceDataset,
    directory: Path,
    feature_version: str,
) -> Path:
    """Write the encoder tables, tagged with the feature definition they belong to.

    ``feature_version`` and ``encoder_digest`` are stored alongside the tables so a loader can
    refuse a table set that does not belong to the model it is about to feed, rather than
    silently encoding against the wrong definition.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = encoder_artifact_path(source_dataset, directory)
    payload = {
        "source_dataset": source_dataset,
        "feature_version": feature_version,
        "encoder_digest": digest_encoders(encoders),
        "columns": list(frequency_columns_for(source_dataset)),
        "encoders": {column: dict(table) for column, table in encoders.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_serving_encoders(
    source_dataset: SourceDataset,
    directory: Path,
    expected_digest: str | None = None,
) -> dict[str, dict[str, float]]:
    """Load the encoder tables for one corpus.

    Args:
        source_dataset: Which corpus's tables to load.
        directory: The artefact directory.
        expected_digest: When given, the ``encoder_digest`` the caller requires. A mismatch is
            refused: encoding against a different table set than the model was fitted on
            changes the decision while leaving every version string looking correct.

    Returns:
        Column name to ``{category: frequency}``.

    Raises:
        FileNotFoundError: If the artefact is absent. The message names the command that
            builds it, because a missing transformer is a setup step rather than a bug.
        ValueError: If ``expected_digest`` does not match.
    """
    path = encoder_artifact_path(source_dataset, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"serving encoders not found at {path}. Build them with:\n"
            f"    python -m app.data.serving_encoders --source-dataset {source_dataset}"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    encoders = {
        column: {category: float(value) for category, value in table.items()}
        for column, table in payload["encoders"].items()
    }
    if expected_digest is not None and payload.get("encoder_digest") != expected_digest:
        raise ValueError(
            f"serving encoders at {path} digest to {payload.get('encoder_digest')!r} but "
            f"{expected_digest!r} was expected. These tables do not belong to the model they "
            "were about to feed."
        )
    return encoders


def _main() -> int:
    """Build and verify the serving encoders from the processed parquet."""
    import argparse

    from app.config import get_settings

    parser = argparse.ArgumentParser(
        description="Rebuild the Phase 1 frequency-encoder tables for serving."
    )
    parser.add_argument("--source-dataset", default="ieee_cis", choices=["ieee_cis", "paysim"])
    arguments = parser.parse_args()
    source: SourceDataset = arguments.source_dataset

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    train_path = settings.processed_data_dir / f"{source}_train.parquet"
    if not train_path.exists():
        logger.error("processed train split not found at %s; run the Phase 1 pipeline", train_path)
        return 1

    columns = list(frequency_columns_for(source))
    generated = [f"freq_{column}" for column in columns]
    frame = pd.read_parquet(train_path, columns=columns + generated)

    encoders = build_from_processed(frame, source)
    verify_against_processed(encoders, frame, source)
    logger.info(
        "rebuilt %d encoder tables and verified them against the pipeline's own output: %s",
        len(encoders),
        ", ".join(f"{column}={len(table)}" for column, table in sorted(encoders.items())),
    )

    # The feature_version the pipeline recorded for this corpus, read from the parquet rather
    # than recomputed, so the artefact is tagged with what actually shipped.
    version_frame = pd.read_parquet(train_path, columns=["feature_version"])
    feature_version = str(version_frame["feature_version"].iloc[0])

    path = save_serving_encoders(encoders, source, settings.artifact_dir, feature_version)
    logger.info("wrote %s (feature_version=%s)", path, feature_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
