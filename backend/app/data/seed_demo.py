"""``python -m app.data.seed_demo`` — populate a fresh database for the dashboard walkthrough.

**What "a clean clone can produce a populated dashboard" can and cannot mean.** The
``transactions`` table is loaded by the Phase 1 pipeline from ``data/raw/train_transaction.csv``
(683MB) and its identity sidecar, both gitignored per the Kaggle terms; ``models/artifacts/``
is gitignored too. No script run against a literal clean clone can produce a populated
dashboard, and this one does not pretend to. What it can do is make one already-set-up
machine's database reproducible in one command, and that is what it does: read the already
processed ``ieee_cis_test.parquet``, load a deterministic sample into ``transactions`` and
``accounts``, then run a subset of that sample through the **real** scoring path so the
decision table, the live feed and the explain drill-down all show real model output.

**Every row is real.** ``is_fraud`` and ``split`` are the corpus's own labels — this table is a
labelled evaluation corpus, not a live ledger, and ``NOT NULL`` on both columns says so.
Fabricating either would poison the same held-out numbers the metrics panel reports elsewhere
in this dashboard. The scored subset's probabilities, ``top_features`` and cost estimates come
from :func:`app.core.serving.score_transaction` — the same function ``POST /score`` calls — so
a fixture that invented plausible-looking SHAP values would be a strictly worse choice than
just running the model.

**Test-split only, and that is a labelling obligation, not a preference.** These rows are the
held-out test set the metrics panel's confusion matrix and PR curve are also computed from.
ml-evaluation-standards item 4.6 forbids showing live-demo output beside held-out numbers
without a label distinguishing the two; the dashboard's provenance chips are that label, and
this script existing does not relieve the frontend of showing them. Train-split rows are never
seeded for a second reason beyond that labelling rule: the model was fitted on them, so a feed
seeded from train would look implausibly clean and would misrepresent what the system does on
data it has not seen.

**Idempotent.** Re-running deletes exactly the rows this script would insert (scoped to
``source_dataset='ieee_cis'`` and the sampled ``transaction_id``s) before inserting them again,
so running it twice leaves one copy, not two.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.core.audit import write_audit_record
from app.core.serving import ModelBundle, score_transaction
from app.data.pipeline import build_accounts
from app.models.account import Account
from app.models.transaction import Transaction

logger = logging.getLogger(__name__)

#: Seeded once, so a re-run samples the same accounts rather than a new random slice each time.
#: Matches the project convention of a logged, fixed seed for every reproducible sampling step.
RANDOM_SEED = 42

#: Accounts with fewer transactions than this have no meaningful history for the velocity and
#: familiarity features to show; accounts with more than this dominate the sample and crowd out
#: variety. Neither bound is load-bearing on correctness -- only on what the demo looks like.
MIN_ACCOUNT_TRANSACTIONS = 2
MAX_ACCOUNT_TRANSACTIONS = 20

DEFAULT_ACCOUNT_SAMPLE = 200
DEFAULT_SCORE_SAMPLE = 300

#: Rounding matches transactions.amount's NUMERIC(20, 4) column.
AMOUNT_QUANTUM = Decimal("0.0001")


def load_test_frame(settings: Settings) -> pd.DataFrame:
    """Return the IEEE-CIS test split the Phase 1 pipeline already wrote.

    Raises:
        SystemExit: With exit code 2, naming the command that produces the file, if it is
            absent. Never silently fabricates a substitute.
    """
    path = settings.processed_data_dir / "ieee_cis_test.parquet"
    if not path.exists():
        raise SystemExit(
            f"seed_demo: {path} does not exist. Run the Phase 1 pipeline first:\n"
            "    python -m app.data.pipeline --source-dataset ieee_cis\n"
            "See data/README.md for the raw download this needs. (exit 2)"
        )
    return pd.read_parquet(path)


def select_demo_sample(
    frame: pd.DataFrame,
    *,
    account_sample: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a deterministic, chronologically-ordered slice of ``frame``.

    Which *accounts* enter the sample is a seeded random choice; the *rows* within each chosen
    account are never reordered, because velocity and familiarity features depend on the order
    transactions actually occurred in.

    Args:
        frame: The full test-split frame.
        account_sample: How many accounts to draw from the eligible pool.
        seed: Fixed so a re-run samples the same accounts.

    Returns:
        The rows belonging to the sampled accounts, sorted by account then event_time.
    """
    counts = frame.groupby("account_id").size()
    eligible = counts[
        (counts >= MIN_ACCOUNT_TRANSACTIONS) & (counts <= MAX_ACCOUNT_TRANSACTIONS)
    ].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("no accounts in the test split fall within the sampling bounds")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(account_sample, len(eligible)), replace=False)

    sample = frame[frame["account_id"].isin(chosen)]
    return sample.sort_values(["account_id", "event_time"], kind="mergesort").reset_index(drop=True)


def select_scoring_subset(
    sample: pd.DataFrame,
    *,
    score_sample: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return the rows to run through the real scoring path, biased toward later-per-account.

    Scoring a transaction requires its account's *prior* rows to already be persisted -- an
    account's first transaction has no history, which is a legitimate cold-start case but not
    an interesting one to show over and over. Preferring the second-or-later row per account,
    then filling any remainder with a seeded random draw, gives the feed and the decision table
    more of the velocity/familiarity features actually firing.
    """
    is_first_per_account = sample.groupby("account_id")["event_time"].cumcount() == 0
    non_first = sample.loc[~is_first_per_account]
    first_rows = sample.loc[is_first_per_account]

    rng = np.random.default_rng(seed)
    if len(non_first) >= score_sample:
        positions = rng.choice(len(non_first), size=score_sample, replace=False)
        chosen: pd.DataFrame = non_first.iloc[np.sort(positions)]
        return chosen

    remaining = score_sample - len(non_first)
    positions = rng.choice(len(first_rows), size=min(remaining, len(first_rows)), replace=False)
    filler = first_rows.iloc[np.sort(positions)]
    combined: pd.DataFrame = pd.concat([non_first, filler]).sort_values(
        ["account_id", "event_time"], kind="mergesort"
    )
    return combined


def to_decimal_amount(value: float) -> Decimal:
    """Convert a parquet float amount to the NUMERIC(20, 4) the table expects."""
    return Decimal(str(value)).quantize(AMOUNT_QUANTUM, rounding=ROUND_HALF_EVEN)


def to_utc_datetime(value: Any) -> datetime:
    """Convert a pandas Timestamp to a timezone-aware ``datetime``."""
    result = pd.Timestamp(value).to_pydatetime()
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def build_transaction_rows(sample: pd.DataFrame) -> list[Transaction]:
    """Build ORM rows for every sampled transaction, carrying the corpus's own labels."""
    rows: list[Transaction] = []
    for record in sample.to_dict(orient="records"):
        device_info = record.get("DeviceInfo")
        addr1 = record.get("addr1")
        rows.append(
            Transaction(
                source_dataset=str(record["source_dataset"]),
                transaction_id=str(record["transaction_id"]),
                event_time=to_utc_datetime(record["event_time"]),
                amount=to_decimal_amount(float(record["amount"])),
                account_id=str(record["account_id"]),
                counterparty_id=None,
                transaction_type=(
                    None
                    if record.get("transaction_type") is None
                    else str(record["transaction_type"])
                ),
                is_fraud=bool(record["is_fraud"]),
                split=str(record["split"]),
                feature_version=str(record["feature_version"]),
                features={},
                device_info=None if device_info is None else str(device_info),
                addr1=None if addr1 is None or pd.isna(addr1) else float(addr1),
            )
        )
    return rows


def build_raw_columns(record: Mapping[Any, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Return the subset of one row's columns that the model's scoring contract accepts.

    Built from whatever the sampled row actually carries, intersected with
    :attr:`app.core.serving.ModelBundle.allowed_raw_columns` -- the same allowlist ``POST
    /score`` validates a caller's ``raw_columns`` against. Values are coerced to JSON-native
    types and non-finite floats become ``None``, matching what a real HTTP client would send.
    """
    raw: dict[str, Any] = {}
    for name in allowed:
        if name not in record:
            continue
        value = record[name]
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if pd.isna(value):
            continue
        if isinstance(value, (np.integer,)):
            raw[name] = int(value)
        elif isinstance(value, (np.floating,)):
            raw[name] = float(value)
        elif isinstance(value, (np.bool_,)):
            raw[name] = bool(value)
        else:
            raw[name] = value
    return raw


async def write_transactions_and_accounts(
    settings: Settings, sample: pd.DataFrame, *, dry_run: bool
) -> None:
    """Delete-then-insert the sampled rows through the read-write pipeline role.

    Uses ``settings.pipeline_url`` (``riskiq_pipeline``), not ``settings.database_url``
    (``riskiq_app``, SELECT-only on these two tables) -- the same distinction
    ``app.data.pipeline.write_postgres`` draws, and for the same reason.
    """
    transaction_ids = sample["transaction_id"].astype(str).tolist()
    accounts = build_accounts(sample, sample["feature_version"].iloc[0])
    account_ids = accounts["account_id"].astype(str).tolist()

    logger.info(
        "seeding %d transactions across %d accounts (dry_run=%s)",
        len(transaction_ids),
        len(account_ids),
        dry_run,
    )
    if dry_run:
        return

    engine = create_async_engine(settings.pipeline_url)
    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session:
            await session.execute(
                delete(Transaction).where(
                    Transaction.source_dataset == "ieee_cis",
                    Transaction.transaction_id.in_(transaction_ids),
                )
            )
            await session.execute(
                delete(Account).where(
                    Account.source_dataset == "ieee_cis",
                    Account.account_id.in_(account_ids),
                )
            )
            session.add_all(build_transaction_rows(sample))
            for record in accounts.to_dict(orient="records"):
                session.add(
                    Account(
                        source_dataset="ieee_cis",
                        account_id=str(record["account_id"]),
                        uid_strategy=str(record["uid_strategy"]),
                        first_seen=to_utc_datetime(record["first_seen"]),
                        last_seen=to_utc_datetime(record["last_seen"]),
                        transaction_count=int(record["transaction_count"]),
                        fraud_count=int(record["fraud_count"]),
                        first_split=str(record["first_split"]),
                        straddles_split=bool(record["straddles_split"]),
                        feature_version=str(record["feature_version"]),
                    )
                )
            await session.commit()
    finally:
        await engine.dispose()


async def score_demo_subset(
    settings: Settings, bundle: ModelBundle, subset: pd.DataFrame, *, dry_run: bool
) -> dict[str, int]:
    """Run each sampled row through the real scoring path, one account-scoped session at a time.

    A fresh session per row, with ``app.current_account_id`` set exactly as
    ``app.db.session.get_scoped_session`` sets it for a live merchant request -- this is
    deliberately not a shortcut around row-level security, even for a seed script running as
    the pipeline role for the bulk load above. The scoring session still connects as
    ``riskiq_app``, matching what a real ``POST /score`` call would use.

    Returns:
        Counts by decision, plus ``degraded`` and ``errors``.
    """
    stats: dict[str, int] = {"allow": 0, "review": 0, "block": 0, "degraded": 0, "errors": 0}
    if dry_run:
        stats["dry_run_rows"] = len(subset)
        return stats

    engine = create_async_engine(settings.database_url)
    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        for record in subset.to_dict(orient="records"):
            account_id = str(record["account_id"])
            raw_columns = build_raw_columns(record, bundle.allowed_raw_columns)
            async with sessionmaker() as session:
                try:
                    await session.execute(
                        text("SELECT set_config('app.current_account_id', :account_id, true)"),
                        {"account_id": account_id},
                    )
                    _outcome, audit_record = await score_transaction(
                        session,
                        bundle,
                        settings,
                        transaction_id=str(record["transaction_id"]),
                        account_id=account_id,
                        event_time=to_utc_datetime(record["event_time"]),
                        amount=to_decimal_amount(float(record["amount"])),
                        raw_columns=raw_columns,
                    )
                    await write_audit_record(session, audit_record)
                    await session.commit()
                    stats[audit_record.decision] = stats.get(audit_record.decision, 0) + 1
                    if audit_record.degraded:
                        stats["degraded"] += 1
                except Exception:
                    await session.rollback()
                    stats["errors"] += 1
                    logger.exception(
                        "scoring failed for %s/%s", account_id, record["transaction_id"]
                    )
    finally:
        await engine.dispose()
    return stats


async def run(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments, seed the corpus rows, then score a subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-sample", type=int, default=DEFAULT_ACCOUNT_SAMPLE)
    parser.add_argument("--score-sample", type=int, default=DEFAULT_SCORE_SAMPLE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be seeded without writing anything.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    frame = load_test_frame(settings)
    sample = select_demo_sample(frame, account_sample=args.account_sample, seed=args.seed)
    await write_transactions_and_accounts(settings, sample, dry_run=args.dry_run)

    scoring_subset = select_scoring_subset(sample, score_sample=args.score_sample, seed=args.seed)

    if args.dry_run:
        logger.info(
            "dry run: would score %d of %d seeded rows",
            len(scoring_subset),
            len(sample),
        )
        print(_summary(sample, scoring_subset, stats={"dry_run_rows": len(scoring_subset)}))
        return 0

    bundle = ModelBundle.load(settings)
    stats = await score_demo_subset(settings, bundle, scoring_subset, dry_run=False)
    print(_summary(sample, scoring_subset, stats=stats))
    return 1 if stats.get("errors", 0) else 0


def _summary(sample: pd.DataFrame, scoring_subset: pd.DataFrame, stats: dict[str, int]) -> str:
    """Render the operator-facing report."""
    lines = [
        "",
        "=== seed_demo summary ===",
        f"accounts seeded:      {sample['account_id'].nunique()}",
        f"transactions seeded:  {len(sample)}",
        f"rows sent to scoring: {len(scoring_subset)}",
    ]
    for key in ("allow", "review", "block", "degraded", "errors"):
        if key in stats:
            lines.append(f"  {key:10s} {stats[key]}")
    if "dry_run_rows" in stats:
        lines.append("(dry run -- nothing was written)")
    lines.append(
        "provenance: real IEEE-CIS test-split rows, real labels, real model output. "
        "Not the held-out evaluation set itself -- see the metrics panel for that."
    )
    return "\n".join(lines)


def main() -> int:
    """Synchronous console-script entry point."""
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
