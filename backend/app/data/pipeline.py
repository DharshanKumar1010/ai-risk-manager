"""The Phase 1 pipeline: load, engineer, split, persist, report.

Run order per corpus, and the order matters:

1. **Load** through the corpus adapter into the canonical frame.
2. **Split** chronologically. This comes *before* feature engineering because the frequency
   encoders must be fitted on the train split only, and a split derived from time is
   independent of any feature, so there is no circularity.
3. **Engineer** features, with encoders fitted on train and applied everywhere.
4. **Persist** to Postgres (canonical columns plus the engineered vector as JSONB) and to
   parquet (the same rows plus the raw model columns, which is what model training reads).
5. **Report** what was produced, including the parts that qualify it.

The two corpora are processed independently end to end. Nothing is concatenated, no split
boundary is shared, and no encoder is fitted across both.

Usage::

    python -m app.data.pipeline --report
    python -m app.data.pipeline --sample 50000 --skip-db      # fast iteration
    python -m app.data.pipeline --sources ieee_cis
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.data import features as feature_engineering
from app.data.adapters import (
    IEEE_CIS_IDENTITY_MODEL_COLUMNS,
    IEEE_CIS_MODEL_COLUMNS,
    AdapterResult,
    iter_ieee_cis_v_block,
    load_ieee_cis,
    load_paysim,
)
from app.data.feature_store import FeatureDefinition, digest_encoders
from app.data.raw_spec import SourceDataset
from app.data.report import (
    IEEE_CIS_COLUMN_GROUPS,
    PAYSIM_COLUMN_GROUPS,
    SourceReport,
    describe_columns,
    missing_value_rates,
    paysim_graph_viability,
    render_report,
    summarise_accounts,
)
from app.data.schema import SPLIT_FRACTIONS, SPLIT_ORDER
from app.data.splitting import assign_splits, find_boundary_overlaps, summarise_splits

logger = logging.getLogger("riskiq.pipeline")

ALL_SOURCES: tuple[SourceDataset, ...] = ("ieee_cis", "paysim")

#: Column order for the COPY into ``transactions``.
TRANSACTION_COPY_COLUMNS: tuple[str, ...] = (
    "source_dataset",
    "transaction_id",
    "event_time",
    "amount",
    "account_id",
    "counterparty_id",
    "transaction_type",
    "is_fraud",
    "split",
    "feature_version",
    "features",
)

ACCOUNT_COPY_COLUMNS: tuple[str, ...] = (
    "source_dataset",
    "account_id",
    "uid_strategy",
    "first_seen",
    "last_seen",
    "transaction_count",
    "fraud_count",
    "first_split",
    "straddles_split",
    "feature_version",
)

#: Rows per COPY batch. Bounds peak memory during record construction; the COPY itself
#: streams either way.
COPY_BATCH_ROWS = 100_000

AMOUNT_QUANTUM = Decimal("0.0001")


@dataclass
class ProcessedSource:
    """One corpus, fully processed and ready to persist."""

    source_dataset: SourceDataset
    frame: pd.DataFrame
    accounts: pd.DataFrame
    definition: FeatureDefinition
    report: SourceReport


def _native_column(series: "pd.Series[Any]") -> list[Any]:
    """Convert a column to JSON-native Python values, nulls included.

    Vectorised deliberately: a per-element loop over ~3.4M rows times ~25 features is tens
    of millions of Python-level operations. Non-finite floats become null rather than
    ``NaN``/``Infinity``, which are not valid JSON and which JSONB rejects.
    """
    if pd.api.types.is_bool_dtype(series):
        return [bool(value) for value in series.to_numpy(dtype=bool)]
    if pd.api.types.is_integer_dtype(series):
        return [int(value) for value in series.to_numpy()]
    if pd.api.types.is_float_dtype(series):
        values = series.to_numpy(dtype="float64")
        boxed = values.astype(object)
        boxed[~np.isfinite(values)] = None
        return list(boxed)
    boxed_object = series.astype(object).where(series.notna(), None)
    return [None if value is None else str(value) for value in boxed_object]


def build_feature_vectors(
    frame: pd.DataFrame, feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Build the per-row engineered feature dictionaries destined for JSONB."""
    columns = {name: _native_column(frame[name]) for name in feature_names}
    names = list(feature_names)
    return [dict(zip(names, row, strict=True)) for row in zip(*columns.values(), strict=True)]


def build_feature_json(frame: pd.DataFrame, feature_names: Sequence[str]) -> list[str]:
    """Serialise the engineered feature vectors for the COPY into ``jsonb``.

    Passed as strings rather than dicts on purpose. ``copy_records_to_table`` uses COPY in
    binary format, and a custom ``set_type_codec`` for jsonb registers a *text* encoder that
    binary COPY cannot use. asyncpg's built-in jsonb codec accepts ``str`` and does have a
    binary encoder, so serialising here is both simpler and the only thing that works.
    """
    return [
        json.dumps(vector, separators=(",", ":"))
        for vector in build_feature_vectors(frame, feature_names)
    ]


def build_accounts(frame: pd.DataFrame, feature_version: str) -> pd.DataFrame:
    """Derive the per-account summary rows from the processed transactions.

    ``first_split`` reads the earliest transaction's split, which is well defined because
    the frame is sorted by ``event_time``. ``straddles_split`` is the leak-surface measure:
    an account whose rows span more than one split.
    """
    grouped = frame.groupby(["source_dataset", "account_id"], sort=False)
    accounts = grouped.agg(
        uid_strategy=("uid_strategy", "first"),
        first_seen=("event_time", "min"),
        last_seen=("event_time", "max"),
        transaction_count=("event_time", "size"),
        fraud_count=("is_fraud", "sum"),
        first_split=("split", "first"),
        distinct_splits=("split", "nunique"),
    ).reset_index()

    accounts["fraud_count"] = accounts["fraud_count"].astype("int32")
    accounts["transaction_count"] = accounts["transaction_count"].astype("int32")
    accounts["straddles_split"] = accounts.pop("distinct_splits") > 1
    accounts["feature_version"] = feature_version
    return accounts


def process_source(
    source_dataset: SourceDataset,
    loaded: AdapterResult,
) -> ProcessedSource:
    """Split, engineer and summarise one corpus.

    Args:
        source_dataset: Which corpus this is.
        loaded: The adapter's canonical frame and notes.

    Returns:
        The processed frame, its derived accounts, its feature definition and its report.
    """
    frame = feature_engineering.sort_for_engineering(loaded.frame)

    split, boundaries = assign_splits(frame["event_time"], source_dataset, SPLIT_FRACTIONS)
    frame["split"] = split
    logger.info(
        "%s: split assigned — %s (boundaries: train < %s, val < %s)",
        source_dataset,
        ", ".join(f"{name}={boundaries.counts[name]:,}" for name in SPLIT_ORDER),
        boundaries.train_end,
        boundaries.val_end,
    )

    frequency_columns = (
        feature_engineering.IEEE_FREQUENCY_COLUMNS
        if source_dataset == "ieee_cis"
        else feature_engineering.PAYSIM_FREQUENCY_COLUMNS
    )
    encoders = feature_engineering.fit_frequency_encoders(
        frame, frame["split"] == "train", frequency_columns
    )
    logger.info(
        "%s: frequency encoders fitted on train only — %s",
        source_dataset,
        ", ".join(f"{column}={len(table)} categories" for column, table in encoders.items()),
    )

    engineered = feature_engineering.engineer_features(
        frame, source_dataset=source_dataset, encoders=encoders
    )

    feature_names = feature_engineering.feature_names_for(source_dataset)
    definition = FeatureDefinition(
        source_dataset=source_dataset,
        feature_names=feature_names,
        parameters={
            **feature_engineering.engineering_parameters(source_dataset),
            "split_fractions": dict(SPLIT_FRACTIONS),
        },
        encoder_digest=digest_encoders(encoders),
    )
    engineered["feature_version"] = definition.feature_version

    accounts = build_accounts(engineered, definition.feature_version)

    positives = int(engineered["is_fraud"].sum())
    logger.info(
        "%s: %d rows, %d positives (%.4f%% base rate), %d accounts, feature_version=%s",
        source_dataset,
        len(engineered),
        positives,
        100.0 * positives / len(engineered) if len(engineered) else 0.0,
        len(accounts),
        definition.feature_version,
    )
    for window in summarise_splits(engineered):
        logger.info(
            "%s/%s: %d rows, %d positives, base rate %.4f%%, %s to %s",
            source_dataset,
            window.split,
            window.rows,
            window.positives,
            100.0 * window.base_rate,
            window.first_event,
            window.last_event,
        )

    groups = IEEE_CIS_COLUMN_GROUPS if source_dataset == "ieee_cis" else PAYSIM_COLUMN_GROUPS
    raw_columns = (
        (*IEEE_CIS_MODEL_COLUMNS, *IEEE_CIS_IDENTITY_MODEL_COLUMNS)
        if source_dataset == "ieee_cis"
        else ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    )

    report = SourceReport(
        source_dataset=source_dataset,
        definition=definition,
        boundaries=boundaries,
        windows=summarise_splits(engineered),
        adapter_notes=loaded.notes,
        missing_rates=missing_value_rates(engineered, groups),
        raw_stats=describe_columns(engineered, ["amount", *raw_columns]).head(20),
        feature_stats=describe_columns(engineered, feature_names),
        account_summary=summarise_accounts(accounts),
    )
    if source_dataset == "paysim":
        report.extras.append(
            ("Tier-3 graph viability (measured, for Phase 4)", paysim_graph_viability(engineered))
        )

    return ProcessedSource(
        source_dataset=source_dataset,
        frame=engineered,
        accounts=accounts,
        definition=definition,
        report=report,
    )


def write_parquet(processed: ProcessedSource, processed_dir: Path) -> list[Path]:
    """Write one parquet file per split.

    Model training reads these rather than Postgres: the same rows, plus the raw model
    columns that the JSONB vector deliberately omits. ``tests/test_pipeline.py`` asserts the
    two stay in agreement on row count, split assignment and feature version.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split in SPLIT_ORDER:
        subset = processed.frame.loc[processed.frame["split"] == split]
        if subset.empty:
            continue
        path = processed_dir / f"{processed.source_dataset}_{split}.parquet"
        subset.to_parquet(path, index=False, compression="snappy")
        written.append(path)
        logger.info("wrote %s (%d rows)", path, len(subset))
    return written


def write_v_block(raw_dir: Path, processed_dir: Path, *, sample: int | None) -> Path | None:
    """Stream IEEE-CIS's ``V1``-``V339`` block to its own parquet file.

    Kept out of the main frame because it is 339 of 394 columns and nothing in Phase 1 reads
    it. Kept rather than discarded because Phase 2 needs it, together with the NaN-pattern
    reduction that has to be fitted on train only.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "ieee_cis_v_block.parquet"
    writer = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for chunk in iter_ieee_cis_v_block(raw_dir, sample=sample):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return None
    logger.info("wrote %s", path)
    return path


def _asyncpg_dsn(database_url: str) -> str:
    """Convert the SQLAlchemy async DSN into the plain form asyncpg accepts."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def write_postgres(processed: ProcessedSource, settings: Settings) -> tuple[int, int]:
    """Replace this corpus's rows in ``transactions`` and ``accounts``.

    Uses asyncpg's binary COPY rather than ORM inserts. At ~3.4M rows an ORM insert would
    take orders of magnitude longer, and COPY composes no SQL string at all — there is no
    injection surface to speak of, which is a stronger position than the parameterised
    statements the security checklist asks for rather than a departure from it.

    The delete is scoped to this corpus and parameterised, so re-running the pipeline for one
    source leaves the other untouched.

    Returns:
        ``(transaction_rows, account_rows)`` written.
    """
    # riskiq_pipeline, not riskiq_app: the application role holds SELECT only on these two
    # tables (security-checklist section 3), and a bulk COPY needs the read-write role.
    connection = await asyncpg.connect(_asyncpg_dsn(settings.pipeline_url))
    try:
        async with connection.transaction():
            await connection.execute(
                "DELETE FROM accounts WHERE source_dataset = $1", processed.source_dataset
            )
            await connection.execute(
                "DELETE FROM transactions WHERE source_dataset = $1", processed.source_dataset
            )

            frame = processed.frame
            feature_names = processed.definition.feature_names
            written = 0
            for start in range(0, len(frame), COPY_BATCH_ROWS):
                chunk = frame.iloc[start : start + COPY_BATCH_ROWS]
                vectors = build_feature_json(chunk, feature_names)
                records = list(
                    zip(
                        chunk["source_dataset"].astype(str),
                        chunk["transaction_id"].astype(str),
                        chunk["event_time"].dt.to_pydatetime(),
                        [
                            Decimal(str(value)).quantize(AMOUNT_QUANTUM)
                            for value in chunk["amount"].to_numpy(dtype="float64")
                        ],
                        chunk["account_id"].astype(str),
                        _native_column(chunk["counterparty_id"]),
                        _native_column(chunk["transaction_type"]),
                        chunk["is_fraud"].to_numpy(dtype=bool).tolist(),
                        chunk["split"].astype(str),
                        chunk["feature_version"].astype(str),
                        vectors,
                        strict=True,
                    )
                )
                await connection.copy_records_to_table(
                    "transactions", records=records, columns=list(TRANSACTION_COPY_COLUMNS)
                )
                written += len(records)
                logger.info(
                    "%s: copied %d/%d transactions", processed.source_dataset, written, len(frame)
                )

            accounts = processed.accounts
            account_records = list(
                zip(
                    accounts["source_dataset"].astype(str),
                    accounts["account_id"].astype(str),
                    accounts["uid_strategy"].astype(str),
                    accounts["first_seen"].dt.to_pydatetime(),
                    accounts["last_seen"].dt.to_pydatetime(),
                    accounts["transaction_count"].astype(int),
                    accounts["fraud_count"].astype(int),
                    accounts["first_split"].astype(str),
                    accounts["straddles_split"].astype(bool),
                    accounts["feature_version"].astype(str),
                    strict=True,
                )
            )
            await connection.copy_records_to_table(
                "accounts", records=account_records, columns=list(ACCOUNT_COPY_COLUMNS)
            )
            logger.info("%s: copied %d accounts", processed.source_dataset, len(account_records))
            return written, len(account_records)
    finally:
        await connection.close()


def load_source(source_dataset: SourceDataset, raw_dir: Path, sample: int | None) -> AdapterResult:
    """Dispatch to the corpus adapter."""
    if source_dataset == "ieee_cis":
        return load_ieee_cis(raw_dir, sample=sample)
    return load_paysim(raw_dir, sample=sample)


async def run_pipeline(
    settings: Settings,
    *,
    sources: Sequence[SourceDataset] = ALL_SOURCES,
    sample: int | None = None,
    skip_db: bool = False,
    skip_v_block: bool = False,
    report_path: Path | None = None,
) -> list[ProcessedSource]:
    """Run the pipeline end to end and write the data-quality report."""
    raw_dir = settings.raw_data_dir
    processed_dir = settings.processed_data_dir
    results: list[ProcessedSource] = []

    for source_dataset in sources:
        logger.info("%s: loading from %s", source_dataset, raw_dir)
        loaded = load_source(source_dataset, raw_dir, sample)
        processed = process_source(source_dataset, loaded)

        overlaps = find_boundary_overlaps(processed.frame)
        if overlaps:
            raise RuntimeError(f"{source_dataset}: split boundaries overlap — {overlaps}")

        write_parquet(processed, processed_dir)
        if source_dataset == "ieee_cis" and not skip_v_block:
            write_v_block(raw_dir, processed_dir, sample=sample)
        if not skip_db:
            await write_postgres(processed, settings)

        results.append(processed)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report([result.report for result in results], sample=sample), encoding="utf-8"
        )
        logger.info("wrote %s", report_path)

    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m app.data.pipeline",
        description="Build the RiskIQ feature store from the raw corpora.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override DATA_DIR.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=list(ALL_SOURCES),
        help="Which corpora to process.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the earliest N rows per corpus (chronological head, not random).",
    )
    parser.add_argument("--skip-db", action="store_true", help="Write parquet only.")
    parser.add_argument(
        "--skip-v-block", action="store_true", help="Skip the IEEE-CIS V1-V339 parquet."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the data-quality report (default: notebooks/eda_report.md).",
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

    report_path = args.report or (settings.reports_dir / "eda_report.md")

    try:
        asyncio.run(
            run_pipeline(
                settings,
                sources=args.sources,
                sample=args.sample,
                skip_db=args.skip_db,
                skip_v_block=args.skip_v_block,
                report_path=report_path,
            )
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
