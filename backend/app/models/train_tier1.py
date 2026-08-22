"""Train, compare and register the Tier-1 layer.

Usage::

    python -m app.models.train_tier1
    python -m app.models.train_tier1 --sources ieee_cis
    python -m app.models.train_tier1 --sample 50000 --skip-registry   # fast iteration

The comparison is the point of this module, so it is worth stating what is being controlled.
Isolation Forest sees no labels; LightGBM sees every one of them. Reporting only that the
supervised model won would be an unsurprising and unfalsifiable claim, and it would confound
two different advantages: supervision, and richer inputs (LightGBM takes native categoricals,
Isolation Forest cannot). So four candidates run per corpus:

===========================  ================================  ===========================
candidate                    matrix                            isolates
===========================  ================================  ===========================
no-skill floor               --                                what PR-AUC means here
amount-only ranking          ``amount_log``                    whether any model earns its keep
isolation forest             numeric-only                      the unsupervised floor
lightgbm (matched)           numeric-only                      supervision alone
lightgbm (full)              numeric + native categoricals     the production candidate
===========================  ================================  ===========================

For PaySim a fifth runs: LightGBM on the ingestion-safe subset, with the post-settlement
balance features removed. The gap between it and the full model is the measured cost of that
corpus recording balances only after a transaction settles.

**The split discipline.** Models fit on train. Early stopping, the operating threshold, the
capacity ceiling, the cost sensitivity sweeps **and the choice of which model ships** all read
validation. Test is scored exactly once, after every one of those decisions is already made,
and only to report. Nothing about test feeds back into any choice — that is what makes the
number reportable at all (ml-evaluation-standards section 1).

The phase brief says to "pick whichever wins on PR-AUC". Applied literally to the test split
that would be a contaminated selection: the winner would have been chosen using the same data
its headline is quoted from. The same criterion is therefore applied one split earlier, on
validation, which is what the brief means rather than what it literally says.

Isolation Forest is kept whatever it scores. It is not a formality: chargeback labels arrive
weeks after the transaction, so the unsupervised floor is the honest answer to "what can we
score on a merchant with no labelled history yet?" — and that is a real operating condition,
not a hypothetical.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.config import Settings, get_settings
from app.data.raw_spec import SourceDataset
from app.data.schema import SPLIT_ORDER, TransactionFeatures
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
    LEAK_SUSPICION_PR_AUC,
    EvaluationResult,
    bootstrap_pr_auc,
    bootstrap_pr_auc_delta,
    evaluate,
    pr_auc,
)
from app.ml.registry import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REGISTRY_PATH,
    RegistryEntry,
    append_entry,
    build_model_id,
)
from app.models.tier1_anomaly import (
    LATENCY_BENCHMARK_CALLS,
    LATENCY_BUDGET_P95_MS,
    ScoreNormaliser,
    Tier1Model,
    benchmark_latency,
)
from app.models.tier1_features import (
    Tier1InputSpec,
    amounts_of,
    fit_tier1_input_spec,
    labels_of,
)

logger = logging.getLogger("riskiq.tier1")

#: Set and logged, per ml-evaluation-standards section 5.
RANDOM_SEED = 42

ALL_SOURCES: tuple[SourceDataset, ...] = ("ieee_cis", "paysim")

#: LightGBM hyperparameters.
#:
#: ``deterministic`` and ``force_row_wise`` together with a fixed ``num_threads`` are what make
#: a re-run reproduce bit-for-bit. Without them LightGBM's multithreaded histogram construction
#: varies between runs, which would quietly falsify the reproducibility claim the registry
#: makes about every model in it.
#:
#: ``average_precision`` is the early-stopping metric because it is the headline metric. Early
#: stopping on logloss would tune the model for a quantity nobody reports.
#:
#: No class reweighting. ``is_unbalance``/``scale_pos_weight`` improve nothing that PR-AUC
#: measures — it is rank-based — while distorting the predicted probabilities that the cost
#: model and Phase 5's meta-learner both consume.
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
    "seed": RANDOM_SEED,
    "verbosity": -1,
}

LIGHTGBM_NUM_ROUNDS = 2_000
LIGHTGBM_EARLY_STOPPING = 100

#: Isolation Forest hyperparameters. ``max_samples=256`` is the value from Liu et al. — the
#: method's whole premise is that anomalies isolate quickly in a *small* subsample, and raising
#: it degrades the score rather than sharpening it.
ISOLATION_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_samples": 256,
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

#: PaySim amounts are simulator units on a scale ~2,500x IEEE-CIS's. Named so no report reads
#: the two corpora's cost figures as the same currency.
PAYSIM_COST_UNITS = "PaySim simulator units (not currency)"
IEEE_COST_UNITS = "IEEE-CIS amount units (consistent with USD)"

#: Ceiling on the share of traffic the capacity-constrained operating point may flag. A stand-in
#: for a real review team's throughput, reported alongside the cost-optimal threshold because
#: that one assumes the queue is unbounded.
MAX_REVIEW_FLAG_RATE = 0.01


@dataclass
class Candidate:
    """One trained model and its measured results."""

    name: str
    model: Tier1Model | None
    validation_threshold: float
    validation_cost: CostEstimate | None
    #: PR-AUC on validation. **This is the model-selection criterion** — see `run_corpus`.
    #: Selecting on the test result instead would make test a validation set and invalidate
    #: the headline (ml-evaluation-standards section 1).
    validation_pr_auc: float
    test_scores: npt.NDArray[np.float64]
    result: EvaluationResult
    notes: list[str] = field(default_factory=list)


@dataclass
class CorpusReport:
    """Everything measured for one corpus."""

    source_dataset: SourceDataset
    spec: Tier1InputSpec
    rows: dict[str, int]
    candidates: list[Candidate]
    winner: Candidate
    runner_up: Candidate
    sensitivity: str
    capacity: EvaluationResult
    latency: dict[str, float]
    delta_interval: tuple[float, float]
    runner_up_interval: tuple[float, float]
    importances: list[tuple[str, float]]
    cost_model: CostModel
    training_window: dict[str, str]

    @property
    def winner_is_significantly_better(self) -> bool:
        """Return whether the winner beats the runner-up beyond sampling noise."""
        return self.runner_up_interval[0] > 0.0


def load_splits(
    processed_dir: Path, source: SourceDataset, sample: int | None
) -> dict[str, pd.DataFrame]:
    """Load the three chronological splits Phase 1 wrote.

    Args:
        processed_dir: ``data/processed``.
        source: Which corpus.
        sample: Keep the earliest N rows of each split, for fast iteration.

    Raises:
        FileNotFoundError: If a split parquet is missing.
        RuntimeError: If the loaded splits are not strictly time-ordered — that would mean the
            files on disk no longer match the Phase 1 guarantee, and every metric derived from
            them would be meaningless.
    """
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        path = processed_dir / f"{source}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m app.data.pipeline` to build the feature store."
            )
        frame = pd.read_parquet(path)
        frames[split] = frame.head(sample) if sample is not None else frame
        logger.info("%s/%s: loaded %d rows from %s", source, split, len(frames[split]), path.name)

    combined = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()], ignore_index=True
    )
    overlaps = find_boundary_overlaps(combined)
    if overlaps:
        raise RuntimeError(f"{source}: split boundaries overlap on disk — {overlaps}")
    return frames


def fit_lightgbm(
    spec: Tier1InputSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    model_id: str,
    numeric_only: bool,
) -> Tier1Model:
    """Fit the supervised candidate, early-stopping on validation average precision."""
    import lightgbm as lgb

    transform = spec.transform_numeric if numeric_only else spec.transform
    train_matrix = transform(train)
    validation_matrix = transform(validation)

    train_set = lgb.Dataset(train_matrix, label=labels_of(train).astype("int8"))
    validation_set = lgb.Dataset(
        validation_matrix, label=labels_of(validation).astype("int8"), reference=train_set
    )
    booster = lgb.train(
        LIGHTGBM_PARAMS,
        train_set,
        num_boost_round=LIGHTGBM_NUM_ROUNDS,
        valid_sets=[validation_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(LIGHTGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    logger.info(
        "%s: lightgbm stopped at %d rounds (best val average_precision %.4f)",
        model_id,
        booster.best_iteration,
        booster.best_score["val"]["average_precision"],
    )
    return Tier1Model(
        model_id=model_id,
        algorithm="lightgbm",
        spec=spec,
        threshold=0.5,  # replaced once the cost-minimising threshold is chosen on validation
        normaliser=ScoreNormaliser(kind="identity"),
        estimator=booster,
        numeric_only=numeric_only,
    )


def fit_isolation_forest(
    spec: Tier1InputSpec,
    train: pd.DataFrame,
    *,
    model_id: str,
) -> Tier1Model:
    """Fit the unsupervised candidate on the numeric-only matrix.

    Fitted on **all** train rows, fraud included. Fitting on known-legitimate rows only would
    read the label and stop being an unsupervised baseline — the whole reason this candidate
    exists is to answer what is achievable before any label arrives.

    ``score_samples`` is used rather than ``predict``: ``contamination`` only sets the offset
    ``predict`` thresholds at, and taking it would substitute a threshold nobody chose on
    validation for the cost-minimising one.
    """
    from sklearn.ensemble import IsolationForest

    # Fitted on the array rather than the frame so the estimator never records feature names.
    # A named estimator later scored on a bare array emits a sklearn warning, and this
    # project's pytest config turns warnings into errors — so scoring and fitting agree on
    # the representation from the start.
    matrix = spec.transform_numeric(train).to_numpy(dtype="float64")
    forest = IsolationForest(**ISOLATION_FOREST_PARAMS).fit(matrix)
    raw = -np.asarray(forest.score_samples(matrix), dtype="float64")
    return Tier1Model(
        model_id=model_id,
        algorithm="isolation_forest",
        spec=spec,
        threshold=0.5,
        normaliser=ScoreNormaliser.fit("minmax", raw),
        estimator=forest,
        numeric_only=True,
    )


def amount_only_scores(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return a ranking by ``amount_log`` alone, min-max scaled into ``[0, 1]``.

    The reference every model must beat to have earned its complexity. It is not a strawman:
    on card fraud, amount alone carries real signal.
    """
    values = frame["amount_log"].to_numpy(dtype="float64")
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = high - low if high > low else 1.0
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / span, 0.0, 1.0)
    return np.asarray(scaled, dtype="float64")


def choose_and_evaluate(
    name: str,
    model: Tier1Model | None,
    validation_scores: npt.NDArray[np.float64],
    test_scores: npt.NDArray[np.float64],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cost_model: CostModel,
    *,
    notes: Sequence[str] = (),
) -> Candidate:
    """Pick the operating threshold on validation, then measure once on test."""
    validation_labels, validation_amounts = labels_of(validation), amounts_of(validation)
    threshold, validation_cost = choose_threshold_by_cost(
        validation_labels, validation_scores, validation_amounts, cost_model
    )

    test_labels, test_amounts = labels_of(test), amounts_of(test)
    result = evaluate(
        name,
        "test",
        test_labels,
        test_scores,
        threshold=threshold,
        threshold_criterion="minimising estimated cost on validation",
        interval=bootstrap_pr_auc(test_labels, test_scores, seed=RANDOM_SEED),
        cost=cost_at_threshold(test_labels, test_scores, test_amounts, threshold, cost_model),
        notes=notes,
    )
    if model is not None:
        model.threshold = threshold
    return Candidate(
        name=name,
        model=model,
        validation_threshold=threshold,
        validation_cost=validation_cost,
        validation_pr_auc=pr_auc(validation_labels, validation_scores),
        test_scores=test_scores,
        result=result,
        notes=list(notes),
    )


def build_scoring_transactions(
    frame: pd.DataFrame,
    model: Tier1Model,
    count: int,
) -> list[TransactionFeatures]:
    """Build assembled scoring contracts from test rows, for the latency benchmark.

    Mirrors what Phase 7 will assemble: the Tier-1 feature vector plus the Tier-1
    ``feature_version``. Non-finite floats become None, as the pipeline does, because that is
    what survives a JSONB round-trip.
    """
    subset = frame.head(count)
    matrix = (
        model.spec.transform_numeric(subset) if model.numeric_only else model.spec.transform(subset)
    )
    transactions: list[TransactionFeatures] = []
    for position in range(len(subset)):
        row = subset.iloc[position]
        vector: dict[str, Any] = {}
        for name in model.feature_names:
            value = matrix.iloc[position][name]
            if isinstance(value, float) and not np.isfinite(value):
                vector[name] = None
            elif pd.isna(value):
                vector[name] = None
            else:
                vector[name] = value if isinstance(value, str) else float(value)
        transactions.append(
            TransactionFeatures(
                transaction_id=str(row["transaction_id"]),
                source_dataset=str(row["source_dataset"]),  # type: ignore[arg-type]
                event_time=row["event_time"].to_pydatetime(),
                amount=Decimal(str(round(float(row["amount"]), 4))),
                account_id=str(row["account_id"]),
                counterparty_id=(
                    None if pd.isna(row["counterparty_id"]) else str(row["counterparty_id"])
                ),
                transaction_type=(
                    None if pd.isna(row["transaction_type"]) else str(row["transaction_type"])
                ),
                feature_version=model.feature_version,
                features=vector,
            )
        )
    return transactions


def run_corpus(
    source: SourceDataset,
    frames: dict[str, pd.DataFrame],
    moment: datetime,
) -> CorpusReport:
    """Fit every candidate for one corpus, compare them, and pick the production model."""
    train, validation, test = frames["train"], frames["val"], frames["test"]
    logger.info("%s: fitting Tier-1 input spec on train only (%d rows)", source, len(train))
    spec = fit_tier1_input_spec(train, source)
    definition = spec.to_feature_definition()
    logger.info(
        "%s: tier1 feature_version=%s, %d features (%d numeric-only)",
        source,
        definition.feature_version,
        len(spec.feature_names),
        len(spec.numeric_feature_names),
    )
    for dropped in spec.dropped:
        logger.info("%s: dropped %s", source, dropped)

    median_amount = float(train["amount"].median())
    cost_model = (
        CostModel(units=IEEE_COST_UNITS)
        if source == "ieee_cis"
        else CostModel.scaled_to(median_amount, PAYSIM_COST_UNITS)
    )
    logger.info(
        "%s: cost model — review=%.2f, chargeback fee=%.2f [%s]",
        source,
        cost_model.review_cost,
        cost_model.chargeback_fee,
        cost_model.units,
    )

    candidates: list[Candidate] = []

    # --- 0b: the reference every model must beat ---------------------------------
    candidates.append(
        choose_and_evaluate(
            "amount-only ranking (reference)",
            None,
            amount_only_scores(validation),
            amount_only_scores(test),
            validation,
            test,
            cost_model,
            notes=["Not a model. The score is `amount_log`, rank-scaled."],
        )
    )

    # --- 1: unsupervised floor ---------------------------------------------------
    forest = fit_isolation_forest(
        spec, train, model_id=build_model_id("tier1-anomaly", "isolation-forest", source, moment)
    )
    candidates.append(
        choose_and_evaluate(
            "IsolationForest (unsupervised)",
            forest,
            forest.score_frame(validation),
            forest.score_frame(test),
            validation,
            test,
            cost_model,
            notes=[
                "Sees no labels. Kept as the cold-start reference: chargeback labels arrive "
                "weeks after the transaction, so this is what is achievable on a merchant "
                "with no labelled history yet.",
            ],
        )
    )

    # --- 2: supervision, on the identical matrix ---------------------------------
    matched = fit_lightgbm(
        spec,
        train,
        validation,
        model_id=build_model_id("tier1-anomaly", "lightgbm-matched", source, moment),
        numeric_only=True,
    )
    candidates.append(
        choose_and_evaluate(
            "LightGBM (matched inputs)",
            matched,
            matched.score_frame(validation),
            matched.score_frame(test),
            validation,
            test,
            cost_model,
            notes=[
                "Fitted on the same numeric-only matrix as the Isolation Forest, so the gap "
                "between the two is supervision alone rather than a difference in inputs.",
            ],
        )
    )

    # --- 3: production candidate --------------------------------------------------
    full = fit_lightgbm(
        spec,
        train,
        validation,
        model_id=build_model_id("tier1-anomaly", "lightgbm", source, moment),
        numeric_only=False,
    )
    candidates.append(
        choose_and_evaluate(
            "LightGBM (full inputs)",
            full,
            full.score_frame(validation),
            full.score_frame(test),
            validation,
            test,
            cost_model,
        )
    )

    # --- 4: PaySim only — what survives without post-settlement balances ----------
    if spec.post_settlement_columns:
        safe_spec = spec.without_post_settlement()
        safe = fit_lightgbm(
            safe_spec,
            train,
            validation,
            model_id=build_model_id("tier1-anomaly", "lightgbm-ingestion-safe", source, moment),
            numeric_only=False,
        )
        candidates.append(
            choose_and_evaluate(
                "LightGBM (ingestion-safe)",
                safe,
                safe.score_frame(validation),
                safe.score_frame(test),
                validation,
                test,
                cost_model,
                notes=[
                    "Post-settlement balance features removed: "
                    + ", ".join(f"`{c}`" for c in spec.post_settlement_columns)
                    + ". A real ingestion-time scorer does not have the balance a "
                    "transaction settles to. The gap against the full model is the measured "
                    "size of that advantage, and this is the deployable number.",
                ],
            )
        )

    # --- Pick the winner on held-out PR-AUC --------------------------------------
    # Selection reads VALIDATION PR-AUC, never the test result. The phase brief says "pick
    # whichever wins on PR-AUC"; doing that on test would make test a validation set and
    # invalidate the headline, so the same criterion is applied one split earlier
    # (ml-evaluation-standards section 1). Test is then scored once, for reporting only.
    trained = [candidate for candidate in candidates if candidate.model is not None]
    ranked = sorted(trained, key=lambda candidate: candidate.validation_pr_auc, reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    baseline = next(
        c for c in trained if c.model is not None and c.model.algorithm == "isolation_forest"
    )
    test_labels = labels_of(test)
    delta_interval = bootstrap_pr_auc_delta(
        test_labels, winner.test_scores, baseline.test_scores, seed=RANDOM_SEED
    )
    # The comparison that decides which model ships, and the one most likely to be an
    # overclaim: the top two candidates are usually close, and picking on the point estimate
    # alone is exactly what the interval machinery exists to prevent.
    runner_up_interval = bootstrap_pr_auc_delta(
        test_labels, winner.test_scores, runner_up.test_scores, seed=RANDOM_SEED
    )
    logger.info(
        "%s: winner is %s (selected on val PR-AUC %.4f; test PR-AUC %.4f)",
        source,
        winner.name,
        winner.validation_pr_auc,
        winner.result.pr_auc,
    )
    if runner_up_interval[0] <= 0.0:
        logger.warning(
            "%s: the advantage of %s over %s is not significant (delta 95%% CI [%.4f, %.4f]). "
            "The simpler candidate performs as well on this test split.",
            source,
            winner.name,
            runner_up.name,
            *runner_up_interval,
        )

    # --- Sensitivity and the capacity-constrained operating point, on validation ---
    assert winner.model is not None
    validation_labels = labels_of(validation)
    validation_amounts = amounts_of(validation)
    winner_validation = winner.model.score_frame(validation)
    sensitivity = "\n\n".join(
        (
            render_sensitivity(
                sensitivity_sweep(
                    validation_labels, winner_validation, validation_amounts, cost_model
                ),
                "Cost sensitivity, +/-50% on BOTH parameters (threshold re-chosen on "
                "validation):",
            ),
            render_sensitivity(
                review_cost_sweep(
                    validation_labels, winner_validation, validation_amounts, cost_model
                ),
                "Cost sensitivity, review cost ONLY (the direction that moves the "
                "recommendation):",
            ),
        )
    )

    # A real review queue has a ceiling; a cost-optimal threshold that ignores it is not
    # deployable however sound its arithmetic. Chosen on validation, measured on test.
    capacity_threshold = threshold_for_flag_rate(winner_validation, MAX_REVIEW_FLAG_RATE)
    capacity = evaluate(
        f"{winner.name} @ capacity",
        "test",
        test_labels,
        winner.test_scores,
        threshold=capacity_threshold,
        threshold_criterion=(
            f"flagging at most {100 * MAX_REVIEW_FLAG_RATE:.1f}% of validation traffic"
        ),
        cost=cost_at_threshold(
            test_labels,
            winner.test_scores,
            amounts_of(test),
            capacity_threshold,
            cost_model,
        ),
        notes=[
            "The same model at a staffable operating point. Reported because the "
            "cost-minimising threshold above assumes unbounded review capacity.",
        ],
    )

    # --- Latency ------------------------------------------------------------------
    transactions = build_scoring_transactions(test, winner.model, count=LATENCY_BENCHMARK_CALLS)
    latency = benchmark_latency(winner.model, transactions)
    logger.info(
        "%s: latency p50=%.3fms p95=%.3fms p99=%.3fms (budget p95 < %.0fms)",
        source,
        latency["p50_ms"],
        latency["p95_ms"],
        latency["p99_ms"],
        LATENCY_BUDGET_P95_MS,
    )

    return CorpusReport(
        source_dataset=source,
        spec=spec,
        rows={split: len(frame) for split, frame in frames.items()},
        candidates=candidates,
        winner=winner,
        runner_up=runner_up,
        sensitivity=sensitivity,
        capacity=capacity,
        latency=latency,
        delta_interval=delta_interval,
        runner_up_interval=runner_up_interval,
        importances=winner.model.top_feature_importances(),
        cost_model=cost_model,
        training_window={
            "start": str(train["event_time"].min()),
            "end": str(train["event_time"].max()),
        },
    )


def register(report: CorpusReport, registry_path: Path, artifact_dir: Path) -> RegistryEntry:
    """Save the winning model and append its record to the registry."""
    winner = report.winner
    assert winner.model is not None
    artifact = winner.model.save(artifact_dir)

    hyperparameters: dict[str, Any] = (
        {
            **LIGHTGBM_PARAMS,
            "num_boost_round": LIGHTGBM_NUM_ROUNDS,
            "early_stopping_rounds": LIGHTGBM_EARLY_STOPPING,
            "best_iteration": int(winner.model.estimator.best_iteration),
        }
        if winner.model.algorithm == "lightgbm"
        else dict(ISOLATION_FOREST_PARAMS)
    )

    # A leak-suspicious result must carry its warning in the permanent record, not only in a
    # log line that scrolls away. Someone reading registry.json alone would otherwise take a
    # PR-AUC of 0.9999 at face value, which is precisely the reading the registry exists to
    # prevent. Prepended so it is the first thing seen.
    caveats: list[str] = []
    if winner.result.is_leak_suspicious:
        caveats.append(
            f"DO NOT QUOTE AS A HEADLINE. Test PR-AUC {winner.result.pr_auc:.4f} exceeds "
            f"{LEAK_SUSPICION_PR_AUC} on fraud data, which ml-evaluation-standards section 4 "
            "treats as a leak signal until disproven. See app/models/README.md and BUILD_LOG.md "
            "for the investigation and the diagnosis before using this number for anything."
        )

    entry = RegistryEntry(
        model_id=winner.model.model_id,
        layer="tier1_anomaly",
        algorithm=winner.model.algorithm,
        source_dataset=report.source_dataset,
        feature_version=winner.model.feature_version,
        training_window=report.training_window,
        hyperparameters=hyperparameters,
        random_seed=RANDOM_SEED,
        heldout_test={
            **winner.result.to_dict(),
            "latency": report.latency,
            "capacity_constrained_operating_point": report.capacity.to_dict(),
        },
        baseline_comparison=[
            candidate.result.to_dict() for candidate in report.candidates if candidate is not winner
        ],
        artifact=str(artifact.relative_to(artifact_dir.parent)),
        notes=[
            *caveats,
            f"Selected on VALIDATION PR-AUC ({winner.validation_pr_auc:.4f}) from "
            f"{len(report.candidates)} candidates; test was scored once afterwards, for "
            "reporting only.",
            f"PR-AUC advantage over the unsupervised Isolation Forest baseline: "
            f"95% CI [{report.delta_interval[0]:.4f}, {report.delta_interval[1]:.4f}].",
            (
                f"Advantage over the runner-up ({report.runner_up.result.model_name}): "
                f"95% CI [{report.runner_up_interval[0]:.4f}, "
                f"{report.runner_up_interval[1]:.4f}]"
            )
            + (
                "."
                if report.winner_is_significantly_better
                else " — includes zero, so this selection is inside sampling noise and the "
                "runner-up is an equally defensible choice."
            ),
            "Top features by split gain: "
            + ", ".join(f"{name} {100 * share:.1f}%" for name, share in report.importances[:5]),
            "Tier-1 mints its own feature_version: it reads the raw row columns the Phase 1 "
            "pipeline kept out of transactions.features, so its input set is genuinely a "
            "different definition from the pipeline's.",
            *winner.notes,
        ],
    )
    append_entry(entry, registry_path)
    logger.info("appended %s to %s", entry.model_id, registry_path)
    return entry


def render_report(reports: Sequence[CorpusReport], sample: int | None) -> str:
    """Render the Phase 2 metrics report."""
    lines = [
        "# RiskIQ — Phase 2 Tier-1 Evaluation",
        "",
        "Generated by `python -m app.models.train_tier1`. Every number is measured on the "
        "held-out **test** split, which was scored exactly once, after the operating threshold "
        "had been chosen on validation.",
        "",
        f"Random seed: `{RANDOM_SEED}`. Latency budget: p95 < {LATENCY_BUDGET_P95_MS:.0f}ms.",
    ]
    if sample is not None:
        lines += [
            "",
            f"> **Sampled run**: only the earliest {sample:,} rows of each split were used. "
            "These numbers are for iteration and are not reportable.",
        ]
    for report in reports:
        winner = report.winner
        lines += [
            "",
            "---",
            "",
            f"## {report.source_dataset}",
            "",
            "### Tier-1 input definition",
            "",
            report.spec.describe(),
            "",
            f"Rows: train {report.rows['train']:,}, val {report.rows['val']:,}, "
            f"test {report.rows['test']:,}.",
            "",
            "### Candidate comparison",
            "",
            "**Selection reads the `val PR-AUC` column, never the test column.** Test is "
            "scored once, after the winner is already chosen, for reporting only — choosing "
            "on it would make it a validation set and invalidate the headline.",
            "",
            "| candidate | val PR-AUC | test PR-AUC | 95% CI | lift | precision | recall | F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        floor = winner.result.base_rate
        lines.append(
            f"| _no-skill floor (random ranker)_ | — | {floor:.4f} | — | 1.0x | — | — | — |"
        )
        for candidate in report.candidates:
            result = candidate.result
            interval = (
                f"{result.pr_auc_interval[0]:.4f}–{result.pr_auc_interval[1]:.4f}"
                if result.pr_auc_interval
                else "—"
            )
            mark = " **(selected)**" if candidate is winner else ""
            validation = (
                f"{candidate.validation_pr_auc:.4f}" if candidate.model is not None else "—"
            )
            lines.append(
                f"| {result.model_name}{mark} | {validation} | {result.pr_auc:.4f} | "
                f"{interval} | {result.lift_over_no_skill:.1f}x | "
                f"{result.confusion.precision:.4f} | "
                f"{result.confusion.recall:.4f} | {result.confusion.f1:.4f} |"
            )
        lines += [
            "",
            f"**Supervision** is worth a PR-AUC delta of 95% CI "
            f"[{report.delta_interval[0]:.4f}, {report.delta_interval[1]:.4f}] over the "
            "unsupervised baseline. "
            + (
                "The interval excludes zero, so the difference is real at this sample size."
                if report.delta_interval[0] > 0
                else "**The interval includes zero — at this sample size the comparison is a "
                "tie, whatever the point estimates suggest.**"
            ),
            "",
            f"**The selected model over the runner-up** "
            f"({report.runner_up.result.model_name}): delta 95% CI "
            f"[{report.runner_up_interval[0]:.4f}, {report.runner_up_interval[1]:.4f}]. "
            + (
                "The interval excludes zero, so the selection is justified."
                if report.winner_is_significantly_better
                else "**The interval includes zero: on this test split the runner-up performs "
                "as well, and the selection rests on a point estimate inside the noise. The "
                "simpler candidate is the defensible choice where the two differ in cost or "
                "complexity.**"
            ),
            "",
            "### What the selected model leans on",
            "",
            "Share of total split gain. A model resting almost entirely on one or two features "
            "is the signature of a dataset artefact rather than a learned pattern.",
            "",
            "| feature | share of gain |",
            "|---|---|",
        ]
        lines += [f"| `{name}` | {100 * share:.1f}% |" for name, share in report.importances[:12]]
        lines += [
            "",
            "### Full results",
            "",
        ]
        for candidate in report.candidates:
            lines += ["```", candidate.result.render(), "```", ""]
        lines += [
            "### Selected model at a staffable operating point",
            "",
            "The threshold above minimises estimated cost, which — with review capacity "
            "assumed unbounded — can imply a queue no team could work. The same model, "
            f"capped at {100 * MAX_REVIEW_FLAG_RATE:.1f}% of traffic:",
            "",
            "```",
            report.capacity.render(),
            "```",
            "",
            "### Cost sensitivity — selected model, thresholds re-chosen on validation",
            "",
            "```",
            report.sensitivity,
            "```",
            "",
            "### Latency — selected model",
            "",
            "```",
            f"{int(report.latency['calls'])} sequential score() calls",
            f"  p50  {report.latency['p50_ms']:.3f} ms",
            f"  p95  {report.latency['p95_ms']:.3f} ms   (budget {LATENCY_BUDGET_P95_MS:.0f} ms)",
            f"  p99  {report.latency['p99_ms']:.3f} ms",
            f"  max  {report.latency['max_ms']:.3f} ms",
            "```",
            "",
            "Measures the scoring call only — vector in, decision out. Feature assembly is "
            "Phase 7's scoring endpoint and is budgeted separately.",
        ]
    return "\n".join(lines) + "\n"


def run(
    settings: Settings,
    *,
    sources: Sequence[SourceDataset] = ALL_SOURCES,
    sample: int | None = None,
    skip_registry: bool = False,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_path: Path | None = None,
) -> list[CorpusReport]:
    """Train and evaluate Tier-1 for each corpus."""
    logger.info("random seed = %d", RANDOM_SEED)
    moment = datetime.now(UTC)
    reports: list[CorpusReport] = []
    for source in sources:
        frames = load_splits(settings.processed_data_dir, source, sample)
        report = run_corpus(source, frames, moment)
        reports.append(report)

        print()
        print(f"================ {source} ================")
        for candidate in report.candidates:
            print()
            print(candidate.result.render())
        print()
        print(report.capacity.render())
        print()
        print(report.sensitivity)
        print()
        print(
            f"Latency ({int(report.latency['calls'])} sequential calls): "
            f"p50={report.latency['p50_ms']:.3f}ms  "
            f"p95={report.latency['p95_ms']:.3f}ms  "
            f"p99={report.latency['p99_ms']:.3f}ms"
        )
        if report.latency["p95_ms"] >= LATENCY_BUDGET_P95_MS:
            logger.error(
                "%s: p95 latency %.3fms exceeds the %.0fms budget",
                source,
                report.latency["p95_ms"],
                LATENCY_BUDGET_P95_MS,
            )
        if report.winner.result.is_leak_suspicious:
            logger.warning(
                "%s: test PR-AUC %.4f exceeds %.2f. Treat as a suspected leak until "
                "disproven (ml-evaluation-standards section 4).",
                source,
                report.winner.result.pr_auc,
                LEAK_SUSPICION_PR_AUC,
            )
        if not skip_registry:
            register(report, registry_path, artifact_dir)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(reports, sample), encoding="utf-8")
        logger.info("wrote %s", report_path)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    """Run Tier-1 training from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m app.models.train_tier1",
        description="Train, compare and register the RiskIQ Tier-1 anomaly layer.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override DATA_DIR.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=list(ALL_SOURCES),
        help="Which corpora to train.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Earliest N rows per split. For iteration only; results are not reportable.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Do not save artefacts or append to models/registry.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the metrics report (default: notebooks/tier1_report.md).",
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
        run(
            settings,
            sources=args.sources,
            sample=args.sample,
            skip_registry=args.skip_registry,
            report_path=args.report or (settings.reports_dir / "tier1_report.md"),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("tier-1 training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
