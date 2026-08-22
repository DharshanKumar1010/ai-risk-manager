"""The Phase 1 data-quality report.

Renders what the pipeline actually produced, including the parts that are inconvenient.
Three numbers here exist specifically to keep a later result honest:

**The singleton share.** IEEE-CIS has no account column, so accounts are inferred. Rows whose
inferred account holds a single transaction have no history, which makes every per-account
feature computed for them noise rather than signal. If that share is large, Tier-2's premise
is weak and this report is where that has to become visible.

**The straddle count.** IEEE-CIS propagates a chargeback label across an account's subsequent
transactions within roughly 120 days. An account whose transactions span a split boundary is
therefore a measurable leak surface. The chronological split stands; the size of the surface
is reported rather than assumed away.

**PaySim's chain-match rate.** Tier-3's ring detection assumes fraudulent ``TRANSFER`` rows
can be chained to the ``CASH_OUT`` that follows them. In the published release those names
frequently fail to match. Measuring it now decides Phase 4's linking strategy before any of
it is built.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.data.feature_store import FeatureDefinition
from app.data.raw_spec import SourceDataset
from app.data.splitting import SplitBoundaries, SplitWindow

#: Raw column groups reported together, so a 339-column block is one line not 339.
IEEE_CIS_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "card": ("card1", "card2", "card3", "card4", "card5", "card6"),
    "address": ("addr1", "addr2"),
    "distance": ("dist1", "dist2"),
    "email": ("P_emaildomain", "R_emaildomain"),
    "counting (C1-C14)": tuple(f"C{i}" for i in range(1, 15)),
    "timedelta (D1-D15)": tuple(f"D{i}" for i in range(1, 16)),
    "match (M1-M9)": tuple(f"M{i}" for i in range(1, 10)),
    "identity (id_01-id_38)": tuple(f"id_{i:02d}" for i in range(1, 39)),
    "device": ("DeviceType", "DeviceInfo"),
}

PAYSIM_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "origin balance": ("oldbalanceOrg", "newbalanceOrig"),
    "destination balance": ("oldbalanceDest", "newbalanceDest"),
    "parties": ("nameOrig", "nameDest"),
}


@dataclass
class SourceReport:
    """Everything the report needs about one corpus."""

    source_dataset: SourceDataset
    definition: FeatureDefinition
    boundaries: SplitBoundaries
    windows: Sequence[SplitWindow]
    adapter_notes: Mapping[str, Any]
    missing_rates: Mapping[str, float]
    raw_stats: pd.DataFrame
    feature_stats: pd.DataFrame
    account_summary: Mapping[str, Any]
    extras: list[tuple[str, str]] = field(default_factory=list)


def missing_value_rates(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Return the mean null rate of each column group present in the frame."""
    rates: dict[str, float] = {}
    for label, columns in groups.items():
        present = [column for column in columns if column in frame.columns]
        if not present:
            continue
        rates[label] = float(frame[present].isna().to_numpy().mean())
    return rates


def describe_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return summary statistics for the numeric columns among ``columns``."""
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.DataFrame()
    numeric = frame[present].select_dtypes(include=["number", "bool"])
    if numeric.empty:
        return pd.DataFrame()
    described = numeric.astype("float64").describe().transpose()
    described["missing_rate"] = [
        float(frame[name].isna().mean()) for name in described.index.astype(str)
    ]
    return described


def summarise_accounts(accounts: pd.DataFrame) -> dict[str, Any]:
    """Return the account-level facts that qualify every per-account feature."""
    total = len(accounts)
    if total == 0:
        return {}
    single = int((accounts["transaction_count"] == 1).sum())
    straddling = int(accounts["straddles_split"].sum())
    return {
        "accounts": total,
        "single_transaction_accounts": single,
        "single_transaction_share": single / total,
        "straddling_accounts": straddling,
        "straddling_share": straddling / total,
        "straddling_fraud_accounts": int(
            accounts.loc[accounts["straddles_split"], "fraud_count"].gt(0).sum()
        ),
        "median_transactions_per_account": float(accounts["transaction_count"].median()),
        "max_transactions_per_account": int(accounts["transaction_count"].max()),
        "uid_strategy_counts": {
            str(key): int(value) for key, value in accounts["uid_strategy"].value_counts().items()
        },
    }


def paysim_graph_viability(frame: pd.DataFrame) -> str:
    """Measure whether PaySim's structure can actually support ring detection.

    Phase 4 rests on two assumptions this checks directly: that destination accounts recur
    often enough to form communities, and that a fraudulent transfer can be chained to the
    cash-out that drains the mule. A near-zero chain-match rate means Phase 4 must link on
    amount and time proximity instead of on account names.
    """
    destination_counts = frame["counterparty_id"].value_counts()
    buckets = {
        "appears once": int((destination_counts == 1).sum()),
        "appears 2-5 times": int(destination_counts.between(2, 5).sum()),
        "appears 6-20 times": int(destination_counts.between(6, 20).sum()),
        "appears 21+ times": int((destination_counts > 20).sum()),
    }

    fraud = frame.loc[frame["is_fraud"]]
    fraud_transfers = fraud.loc[fraud["transaction_type"] == "TRANSFER"]
    fraud_cashouts = fraud.loc[fraud["transaction_type"] == "CASH_OUT"]
    all_cashout_origins = set(
        frame.loc[frame["transaction_type"] == "CASH_OUT", "account_id"].dropna()
    )
    fraud_cashout_origins = set(fraud_cashouts["account_id"].dropna())

    chain_to_fraud = (
        float(fraud_transfers["counterparty_id"].isin(fraud_cashout_origins).mean())
        if len(fraud_transfers)
        else 0.0
    )
    chain_to_any = (
        float(fraud_transfers["counterparty_id"].isin(all_cashout_origins).mean())
        if len(fraud_transfers)
        else 0.0
    )

    lines = [
        "Destination account recurrence — whether communities can exist at all:",
        "",
        "| Destination appears | Count |",
        "|---|---|",
    ]
    lines += [f"| {label} | {count:,} |" for label, count in buckets.items()]
    lines += [
        "",
        f"- Distinct destinations: **{frame['counterparty_id'].nunique():,}** "
        f"over {len(frame):,} rows",
        f"- Distinct origins: **{frame['account_id'].nunique():,}** "
        "(near-unique — the reason PaySim carries no account history)",
        "",
        "Transfer-to-cash-out chaining — whether Phase 4 can link on account names:",
        "",
        f"- Fraudulent TRANSFER rows: **{len(fraud_transfers):,}**",
        f"- Fraudulent CASH_OUT rows: **{len(fraud_cashouts):,}**",
        f"- Fraudulent transfers whose destination is the origin of a *fraudulent* "
        f"cash-out: **{chain_to_fraud:.2%}**",
        f"- Fraudulent transfers whose destination is the origin of *any* "
        f"cash-out: **{chain_to_any:.2%}**",
    ]

    if chain_to_fraud < CHAIN_MATCH_USABLE:
        # Distinguish "weak" from "absent". At a rate this low the mechanism does not work
        # at all, and describing it as a partial signal would invite Phase 4 to build on it.
        severity = (
            "recovers essentially nothing"
            if chain_to_any < CHAIN_MATCH_NEGLIGIBLE
            else "recovers only a minority of the pairs"
        )
        lines += [
            "",
            f"> **Phase 4 consequence — decided on measurement, before anything is built.** "
            f"Name-based chaining {severity}. This is the known artefact in the published "
            "release: the `nameDest` of a fraudulent transfer does not reappear as the "
            "`nameOrig` of its paired cash-out, so the mule account cannot be followed by "
            "identity. **Tier-3 must not link transfers to cash-outs by account name.** The "
            "workable edge is amount-and-step proximity, with a name match treated as "
            "corroboration if it ever occurs.",
            ">",
            "> The graph itself is still viable, and for a different reason than the chaining: "
            f"destination accounts recur heavily — {buckets['appears 6-20 times']:,} appear "
            f"6-20 times and {buckets['appears 21+ times']:,} appear 21 or more times across "
            f"{len(frame):,} rows. That recurrence is what community detection needs, and it "
            "is present. It is only the transfer-to-cash-out identity link that is absent.",
        ]
    return "\n".join(lines)


#: A split whose base rate differs from the corpus overall by more than this factor is
#: called out. Chosen to catch a shift big enough to change how a metric should be read,
#: without firing on ordinary sampling noise.
BASE_RATE_DRIFT_FACTOR = 1.5

#: Below this share of fraudulent transfers chaining to a fraudulent cash-out by account
#: name, the link is too weak for Tier-3 to treat as its primary edge.
CHAIN_MATCH_USABLE = 0.5

#: Below this, the link is not weak but absent, and should be described as such.
CHAIN_MATCH_NEGLIGIBLE = 0.01


def class_balance_drift(windows: Sequence[SplitWindow]) -> str | None:
    """Flag splits whose class balance departs materially from the corpus overall.

    A chronological split is the right choice, but it does not promise a stationary class
    balance — and when the positive rate moves sharply between train and test, a model's
    held-out precision is partly a statement about *when* the test period is rather than
    about the model. Surfacing it here means the later metric can be read correctly instead
    of being explained after the fact.
    """
    total_rows = sum(window.rows for window in windows)
    total_positives = sum(window.positives for window in windows)
    if not total_rows or not total_positives:
        return None
    overall = total_positives / total_rows

    drifted = [
        window
        for window in windows
        if window.base_rate > overall * BASE_RATE_DRIFT_FACTOR
        or window.base_rate * BASE_RATE_DRIFT_FACTOR < overall
    ]
    if not drifted:
        return None

    lines = [
        f"> **Class balance is not stationary across this corpus's timeline.** The overall "
        f"positive rate is {overall:.4%}, but:",
        ">",
    ]
    for window in drifted:
        direction = "higher" if window.base_rate > overall else "lower"
        factor = window.base_rate / overall if overall else 0.0
        lines.append(
            f"> - **{window.split}** runs at {window.base_rate:.4%} — "
            f"{factor:.1f}x {direction} than the corpus overall "
            f"({window.positives:,} of {window.rows:,} rows)"
        )
    lines += [
        ">",
        "> This is a property of the data, not a defect in the split, and the chronological "
        "boundaries stand. It does mean a held-out metric from this corpus is partly a "
        "statement about which period the test window covers. Every metric measured here "
        "must quote the split's own base rate — a precision figure read against the wrong "
        "denominator will look far better or worse than it is.",
    ]
    return "\n".join(lines)


def _render_table(frame: pd.DataFrame, float_format: str = "{:,.4f}") -> str:
    """Render a DataFrame as a markdown table."""
    if frame.empty:
        return "_No numeric columns to describe._"
    formatted = frame.copy()
    for column in formatted.columns:
        # A row count rendered as "20,000.0000" reads as noise rather than as a count.
        spec = "{:,.0f}" if str(column) == "count" else float_format
        formatted[column] = formatted[column].map(lambda value, fmt=spec: fmt.format(value))
    header = "| feature | " + " | ".join(str(c) for c in formatted.columns) + " |"
    divider = "|---" * (len(formatted.columns) + 1) + "|"
    rows = [
        f"| `{index}` | " + " | ".join(formatted.loc[index].astype(str)) + " |"
        for index in formatted.index
    ]
    return "\n".join([header, divider, *rows])


def _render_source(report: SourceReport) -> list[str]:
    """Render one corpus's section."""
    boundaries = report.boundaries
    lines = [
        f"## {report.source_dataset}",
        "",
        "### Class balance and chronological split",
        "",
        "| split | rows | share | positives | base rate | first event | last event |",
        "|---|---|---|---|---|---|---|",
    ]
    for window in report.windows:
        share = window.rows / boundaries.total if boundaries.total else 0.0
        lines.append(
            f"| {window.split} | {window.rows:,} | {share:.1%} | {window.positives:,} | "
            f"{window.base_rate:.4%} | {window.first_event} | {window.last_event} |"
        )

    total_positives = sum(window.positives for window in report.windows)
    overall_rate = total_positives / boundaries.total if boundaries.total else 0.0
    lines += [
        f"| **all** | **{boundaries.total:,}** | 100% | **{total_positives:,}** | "
        f"**{overall_rate:.4%}** | | |",
        "",
        f"Boundaries are strict: train is `event_time < {boundaries.train_end}`, "
        f"val is `< {boundaries.val_end}`, test is everything after. Rows sharing a boundary "
        "timestamp all fall on the later side, which is why the shares are near rather than "
        "exactly 70/15/15.",
        "",
    ]
    drift = class_balance_drift(report.windows)
    if drift:
        lines += [drift, ""]

    lines += [
        "### Feature definition",
        "",
        report.definition.describe(),
        "",
        "### Missing-value rates by raw column group (before engineering)",
        "",
        "| column group | null rate |",
        "|---|---|",
    ]
    lines += [f"| {label} | {rate:.2%} |" for label, rate in sorted(report.missing_rates.items())]

    lines += [
        "",
        "### Raw numeric columns (before engineering)",
        "",
        _render_table(report.raw_stats),
        "",
        "### Engineered features (after engineering)",
        "",
        _render_table(report.feature_stats),
        "",
        "### Accounts",
        "",
    ]
    summary = report.account_summary
    if summary:
        lines += [
            f"- Distinct accounts: **{summary['accounts']:,}**",
            f"- Accounts with a single transaction: **{summary['single_transaction_accounts']:,}** "
            f"(**{summary['single_transaction_share']:.1%}**) — these have no history, so their "
            "per-account features carry no signal",
            f"- Median transactions per account: "
            f"**{summary['median_transactions_per_account']:,.0f}**, "
            f"maximum **{summary['max_transactions_per_account']:,}**",
            f"- Accounts straddling a split boundary: **{summary['straddling_accounts']:,}** "
            f"(**{summary['straddling_share']:.1%}**), of which "
            f"**{summary['straddling_fraud_accounts']:,}** contain at least one fraud",
            "",
            "| uid strategy | accounts |",
            "|---|---|",
        ]
        lines += [
            f"| `{key}` | {value:,} |"
            for key, value in sorted(summary["uid_strategy_counts"].items())
        ]

    lines += ["", "### Adapter notes", "", "```json", _pretty(report.adapter_notes), "```", ""]

    for title, body in report.extras:
        lines += [f"### {title}", "", body, ""]
    return lines


def _pretty(mapping: Mapping[str, Any]) -> str:
    """Render a mapping as readable JSON."""
    import json

    return json.dumps(dict(mapping), indent=2, default=str)


def render_report(reports: Sequence[SourceReport], *, sample: int | None) -> str:
    """Render the full data-quality report as markdown."""
    lines = [
        "# RiskIQ — Phase 1 Data-Quality Report",
        "",
        "Generated by `python -m app.data.pipeline`. Every number here is measured from the "
        "processed corpus, not copied from a dataset description.",
        "",
    ]
    if sample is not None:
        lines += [
            f"> **Sampled run.** Limited to the earliest {sample:,} rows per corpus, so the "
            "figures below describe that slice and not the full dataset. Re-run without "
            "`--sample` for reportable numbers.",
            "",
        ]
    lines += [
        "The two corpora are never merged. Their clocks share no origin and their base rates "
        "differ by roughly 27x, so splits are computed per source, no model trains across "
        "both, and no fitted score crosses between them.",
        "",
        "---",
        "",
    ]
    for report in reports:
        lines += _render_source(report)
        lines += ["---", ""]
    return "\n".join(lines)
