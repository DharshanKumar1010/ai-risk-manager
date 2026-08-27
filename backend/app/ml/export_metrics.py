"""``python -m app.ml.export_metrics`` — write the dashboard's held-out metrics as a TS module.

**Why a build-time export rather than a `GET /metrics` endpoint.** Three reasons, in order of
weight. First, ``models/registry.json`` carries ``pr_auc`` and its confidence interval but no
precision/recall *points* -- the PR curve the phase brief names first has to be recomputed from
the Tier-1 booster and the test parquet, which takes the model artefacts and ~30 seconds; that
is not request-time work on any host, let alone a free-tier Render dyno with neither the
parquet nor the memory to read it per request. Second, the frontend (Vercel) and the backend
(Render) are separate deployments; a live endpoint would make the dashboard's strongest panel
depend on a warm backend on every visit, and show nothing at all during a cold start. Third,
and the reason this module writes TypeScript rather than JSON: ``resolveJsonModule`` would let
the frontend import a JSON literal with no field ever required, so a future edit to this script
could silently stop emitting ``false_positive_cost.assumptions`` and nothing would fail until a
human noticed the panel had gone quiet. Emitting against a hand-written interface
(``frontend/src/types/metrics.ts``) that marks the ml-evaluation-standards fields as
**required** makes ``tsc -b --noEmit`` -- already a CI gate via ``npm run lint`` -- refuse a
build that ever drops one.

**What this cannot check.** ``--check`` mode regenerates the registry-derived block and diffs
it against the committed file, catching drift between ``registry.json`` and what the frontend
ships. It cannot regenerate the PR curve in CI, because CI has neither the parquet nor the
Tier-1 artefact (both gitignored). That is a real gap, not an oversight: state it plainly
rather than let ``--check`` imply coverage it does not have.

**Provenance travels with every number.** Every exported block carries the registry
``model_id`` it came from and this export's own timestamp, so the panel can render "held-out
test · tier1-anomaly-lightgbm-... · exported 2026-08-26" rather than a bare figure — the
provenance chip ml-evaluation-standards item 4.6 requires to distinguish held-out numbers from
anything live-demo.

**No operating threshold or per-unit cost figure survives this export, anywhere.** This file
is inlined into a public, unauthenticated JS bundle -- unlike every API response, which
``schemas.py`` already withholds ``risk_band``/probability/cost fields from for the same
reason. The registry's raw ``threshold``, ``threshold_criterion``, per-policy and per-curve-
point thresholds, and ``review_cost_per_false_positive``/``flat_chargeback_fee_per_false_
negative`` are all scrubbed before anything is written -- see ``_scrub_false_positive_cost``'s
docstring for the exploit this closes. What ships instead: PR-AUC and its CI, precision/
recall/F1, the full confusion matrix, the base rate, and the false-positive cost *totals* --
everything ml-evaluation-standards requires, none of it the live decision boundary.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.ml.registry import latest_entry, read_registry
from app.models.tier1_anomaly import Tier1Model

logger = logging.getLogger(__name__)

#: How many points the PR curve is downsampled to. The raw curve has one point per distinct
#: threshold (tens of thousands on 88,581 rows); a chart does not need more than this to read
#: correctly, and the file stays small enough to ship in a JS bundle.
PR_CURVE_POINTS = 200

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3] / "frontend" / "src" / "data" / "metrics.generated.ts"
)


def select_registry_blocks(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return the latest IEEE-CIS entry for each layer the dashboard reports on.

    Only what the panel actually renders survives -- not the full registry entry.
    Hyperparameters, superseded tier3 snapshots and the loss model's 113 feature names are
    dropped; a metrics panel has no use for them and they are most of the 441KB source file.
    """
    registry_entries = read_registry(settings.registry_path)
    blocks: dict[str, dict[str, Any]] = {}
    for layer, key in (
        ("tier1_anomaly", "tier1"),
        ("causal_cost", "causal_cost"),
        ("meta_learner", "meta_learner"),
        ("tier3_graph", "tier3"),
    ):
        entry = latest_entry(layer, "ieee_cis", settings.registry_path)
        if entry is None:
            logger.warning("no %s/ieee_cis entry in the registry; %s block omitted", layer, key)
            continue
        blocks[key] = entry
    if len(registry_entries) == 0:
        raise RuntimeError(f"registry at {settings.registry_path} is empty")
    return blocks


#: Assumption sentences that state a literal per-unit cost figure -- see
#: ``_scrub_false_positive_cost``'s docstring for why these, specifically, cannot ship.
_UNIT_COST_ASSUMPTION_PREFIXES = ("A false positive costs", "A false negative costs")


def _scrub_false_positive_cost(cost: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip the operating threshold and the per-unit cost figures from a cost block.

    **Why this exists, found in security review rather than designed in up front.** The raw
    registry entry's ``false_positive_cost`` carries the exact live operating ``threshold``
    alongside ``review_cost_per_false_positive``/``flat_chargeback_fee_per_false_negative``.
    ``app/models/causal_cost.py`` states the served policy's decision rule in closed form:
    ``allow`` iff ``p <= r / (A + f + r)``. Shipping the threshold and both unit costs in a
    public, unauthenticated static bundle hands any visitor the exact decision boundary with
    zero probing -- strictly worse than the residual decision-oracle exposure
    ``schemas.py``'s module docstring already accepts as bounded by auth and rate limiting,
    because this needs no requests against the live API at all. ``risk_band`` was removed from
    every API response for exactly this reason; this closes the same hole reopened by a
    different channel.

    What survives: the total dollar cost, the false positive/negative *counts*, and every
    generic methodological caveat ("cost is linear in the number of mistakes", "review capacity
    is unbounded"). Those two counts and the total technically let a reader back out the unit
    costs by division -- accepted, because doing so still never yields the threshold, and the
    threshold is the one number the break-even formula needs as its anchor.
    """
    if cost is None:
        return None
    scrubbed = dict(cost)
    scrubbed.pop("threshold", None)
    scrubbed.pop("review_cost_per_false_positive", None)
    scrubbed.pop("flat_chargeback_fee_per_false_negative", None)
    assumptions = scrubbed.get("assumptions")
    if isinstance(assumptions, list):
        scrubbed["assumptions"] = [
            text
            for text in assumptions
            if not (isinstance(text, str) and text.startswith(_UNIT_COST_ASSUMPTION_PREFIXES))
        ]
    return scrubbed


def _scrub_operating_point(point: dict[str, Any] | None) -> dict[str, Any] | None:
    """Apply the same scrub to a capacity-constrained operating point block.

    Same shape as the top-level ``heldout_test`` entry -- its own ``threshold``,
    ``threshold_criterion`` and nested ``false_positive_cost`` -- so it needs the same
    treatment, not a separate hole left open one level down.
    """
    if point is None:
        return None
    scrubbed = dict(point)
    scrubbed.pop("threshold", None)
    scrubbed.pop("threshold_criterion", None)
    scrubbed["false_positive_cost"] = _scrub_false_positive_cost(
        scrubbed.get("false_positive_cost")
    )
    return scrubbed


def compute_pr_curve(settings: Settings, tier1_entry: dict[str, Any]) -> dict[str, Any]:
    """Recompute the PR curve from the shipped Tier-1 model and the held-out test split.

    Requires the model artefact and the processed parquet, neither of which CI has -- this is
    the half of the export ``--check`` cannot verify. See the module docstring.
    """
    from sklearn.metrics import precision_recall_curve

    model_id = str(tier1_entry["model_id"])
    feature_version = str(tier1_entry.get("feature_version", "")) or None
    model = Tier1Model.load(model_id, settings.artifact_dir, feature_version=feature_version)

    test_path = settings.processed_data_dir / "ieee_cis_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} does not exist; run the Phase 1 pipeline before exporting the PR curve"
        )
    frame = pd.read_parquet(test_path)
    frame = frame[frame["split"] == "test"].reset_index(drop=True)

    scores = model.score_frame(frame)
    labels = frame["is_fraud"].to_numpy(dtype=bool)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more precision/recall point than thresholds (the
    # final point is precision=1, recall=0 with no corresponding threshold). Drop it so every
    # point in the export has a defined operating threshold.
    precision, recall = precision[:-1], recall[:-1]

    if len(thresholds) > PR_CURVE_POINTS:
        # Downsample by recall rather than by array position: recall is monotonically
        # non-increasing along this curve, so an even stride in *index* over-represents the
        # dense, near-zero-threshold tail. Evenly spaced positions still read correctly on a
        # recall-vs-precision chart because the ordering is preserved either way.
        positions = np.linspace(0, len(thresholds) - 1, PR_CURVE_POINTS).astype(int)
    else:
        positions = np.arange(len(thresholds))

    return {
        # No `threshold` per point, deliberately -- see `_scrub_false_positive_cost`'s
        # docstring. The chart only ever plots precision against recall; a threshold per point
        # would let a reader read the exact score cutoff for any point on the curve straight
        # off the bundle.
        "points": [
            {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
            }
            for i in positions
        ],
        "base_rate": round(float(labels.mean()), 6),
        "n": int(len(labels)),
        "positives": int(labels.sum()),
    }


def build_export(settings: Settings, *, include_pr_curve: bool) -> dict[str, Any]:
    """Assemble the full exported payload."""
    blocks = select_registry_blocks(settings)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_sha256": _registry_digest(settings.registry_path),
        "tier1": _tier1_block(blocks.get("tier1")),
        "causal_cost": _causal_cost_block(blocks.get("causal_cost")),
        "meta_learner": _meta_learner_block(blocks.get("meta_learner")),
        "tier3": _tier3_block(blocks.get("tier3")),
    }
    if include_pr_curve and "tier1" in blocks:
        payload["tier1"]["pr_curve"] = compute_pr_curve(settings, blocks["tier1"])
    return payload


def _tier3_block(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Tier-3's ring-level held-out result -- the number `RingsPanel` displays a graph for
    without ever quoting.

    Read from ``heldout_test.ring_level``, not the top-level ``heldout_test`` block: this
    corpus's *unit of analysis is the ring*, not the transaction (``unit_of_analysis`` in the
    same entry says so, and ``notebooks/tier3_report.md`` explains why -- IEEE-CIS accounts
    barely recur, so a per-transaction ring score abstains on 65%+ of test rows and is
    separately reported, elsewhere, as below its own no-skill floor). Carrying ``unit_of_
    analysis`` through as a literal is deliberate: a reader placing this PR-AUC next to
    Tier-1's must not be able to mistake the two for the same measurement on the same units.
    """
    if entry is None:
        return None
    ring = entry["heldout_test"]["ring_level"]
    return {
        "model_id": entry["model_id"],
        "unit_of_analysis": "ring",
        "rows": ring["rows"],
        "positives": ring["positives"],
        "base_rate": ring["base_rate"],
        "pr_auc": ring["pr_auc"],
        "pr_auc_ci95": ring["pr_auc_ci95"],
        "pr_auc_no_skill_floor": ring["pr_auc_no_skill_floor"],
        "confusion_matrix": ring["confusion_matrix"],
        "false_positive_cost": _scrub_false_positive_cost(ring["false_positive_cost"]),
    }


def _registry_digest(path: Path) -> str:
    """Hash the registry file's bytes, so a consumer can tell which registry state this
    export was taken from without trusting the timestamp alone."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _tier1_block(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    heldout = entry["heldout_test"]
    return {
        "model_id": entry["model_id"],
        "feature_version": entry.get("feature_version"),
        "rows": heldout["rows"],
        "positives": heldout["positives"],
        "base_rate": heldout["base_rate"],
        "pr_auc": heldout["pr_auc"],
        "pr_auc_ci95": heldout["pr_auc_ci95"],
        "pr_auc_no_skill_floor": heldout["pr_auc_no_skill_floor"],
        "precision": heldout["precision"],
        "recall": heldout["recall"],
        "f1": heldout["f1"],
        "confusion_matrix": heldout["confusion_matrix"],
        "false_positive_cost": _scrub_false_positive_cost(heldout["false_positive_cost"]),
        "capacity_constrained_operating_point": _scrub_operating_point(
            heldout.get("capacity_constrained_operating_point")
        ),
    }


def _causal_cost_block(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    heldout = entry["heldout_test"]
    notes = [
        note
        for note in entry.get("notes", [])
        if not (isinstance(note, str) and note.startswith("Break-even probability"))
    ]
    return {
        "model_id": entry["model_id"],
        "rows": heldout["rows"],
        "positives": heldout["positives"],
        "base_rate": heldout["base_rate"],
        "confusion_matrix": heldout["confusion_matrix"],
        "false_positive_cost": _scrub_false_positive_cost(heldout["false_positive_cost"]),
        "cost_reduction_vs_baseline": heldout["cost_reduction_vs_baseline"],
        "policies": _policy_summaries(heldout.get("policies")),
        "calibration": heldout.get("calibration"),
        "sensitivity": _scrub_sensitivity(heldout.get("sensitivity")),
        "cnp_regime": _strip_thresholds(heldout.get("cnp_regime")),
        "cnp_regime_deltas": heldout.get("cnp_regime_deltas"),
        "ope_validation_caveat": _first_ope_caveat(heldout.get("ope_validation")),
        "notes": notes,
    }


def _scrub_sensitivity(rows: Any) -> list[dict[str, Any]] | None:
    """Drop the swept threshold from each cost-sensitivity row.

    The "1x" row is the actual shipped cost model at its actual operating point in different
    clothing -- the same threshold this whole export otherwise withholds, so the sweep cannot
    carry it either. ``factor``, ``flag_rate`` and ``total_cost`` are what the panel's table
    renders and are unaffected.
    """
    return _strip_thresholds(rows)


def _strip_thresholds(rows: Any) -> list[dict[str, Any]] | None:
    """Drop the ``threshold`` key from a list of dict rows, unconditionally.

    Shared by every raw registry passthrough that carries a per-row operating threshold
    (``cnp_regime``, the sensitivity sweep) -- see ``_scrub_false_positive_cost``'s docstring
    for why this export withholds the number everywhere it appears, not just at the one place
    a reviewer happened to look first.
    """
    if not isinstance(rows, list):
        return None
    scrubbed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        row.pop("threshold", None)
        scrubbed.append(row)
    return scrubbed


def _policy_summaries(policies: Any) -> list[dict[str, Any]] | None:
    """Trim the three ranking strategies to what the default-regime cost chart needs.

    Mirrors the shape ``cnp_regime`` already carries (one row per policy, at the shipped
    operating point), so the same chart component can render both regimes from one prop
    shape without a separate code path for "the default regime happens to store this
    differently".
    """
    if not isinstance(policies, list):
        return None
    summaries = []
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        shipped = policy.get("shipped_point", {})
        summaries.append(
            {
                "policy": policy.get("strategy"),
                "flag_rate": shipped.get("flag_rate"),
                "precision": shipped.get("precision"),
                "recall": shipped.get("recall"),
                "recall_by_value": shipped.get("recall_by_value"),
                "caught_fraud_value": shipped.get("caught_fraud_value"),
                "missed_fraud_value": shipped.get("missed_fraud_value"),
                "confusion_matrix": shipped.get("confusion_matrix"),
                "cost_per_1000": shipped.get("cost_per_1000"),
            }
        )
    return summaries


def _first_ope_caveat(ope_validation: Any) -> str | None:
    """Return the caveat string every ``ope_validation`` row repeats.

    ``ope_validation`` is a *list* of rows, one per policy, each carrying its own ``caveat``
    field with the same text ("the logging policy is simulated..."). One copy is enough for
    the panel; the list itself is not carried into the export.
    """
    if not isinstance(ope_validation, list) or not ope_validation:
        return None
    first = ope_validation[0]
    if not isinstance(first, dict):
        return None
    caveat = first.get("caveat")
    return str(caveat) if caveat is not None else None


def _meta_learner_block(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """The retired layer's own numbers -- see BUILD_LOG.md: shipped anyway is not this."""
    if entry is None:
        return None
    heldout = entry["heldout_test"]
    return {
        "model_id": entry["model_id"],
        "pr_auc": heldout["pr_auc"],
        "pr_auc_ci95": heldout["pr_auc_ci95"],
        "delta_vs_tier1_alone": heldout.get("delta_vs_tier1_alone"),
        "notes": entry.get("notes", []),
    }


def render_typescript(payload: dict[str, Any]) -> str:
    """Render the payload as a typed TS module.

    The declared type is an ANNOTATION, not an `as` assertion -- deliberately. An assertion
    would suppress exactly the checks this file exists to enforce: TypeScript validates an
    `as T` cast only for loose overlap between the literal's inferred type and T, which admits
    a payload missing a required field. A plain annotation on a literal assigned directly
    performs full structural checking, so every field ml-evaluation-standards requires
    alongside a headline number -- required, not optional, on ``HeldoutMetrics`` in
    ``types/metrics.ts`` -- makes a future export that drops one fail ``tsc -b --noEmit``,
    which ``npm run lint`` already runs in CI.
    """
    body = json.dumps(payload, indent=2, sort_keys=True)
    return (
        "// AUTOGENERATED by `python -m app.ml.export_metrics`. Do not edit by hand --\n"
        "// re-run the exporter after registry.json changes.\n"
        "//\n"
        "// Every field ml-evaluation-standards requires alongside a headline number is\n"
        "// REQUIRED on HeldoutMetrics in types/metrics.ts, not optional -- a future export\n"
        "// that drops one fails `tsc -b --noEmit`, which is already a CI gate. This is a type\n"
        "// ANNOTATION below, not an `as` assertion -- an assertion would suppress that check.\n"
        "import type { GeneratedMetrics } from '@/types/metrics'\n\n"
        f"export const METRICS: GeneratedMetrics = {body}\n"
    )


def run(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate the registry-derived block and diff against the committed file. "
        "Cannot verify the PR curve -- see the module docstring.",
    )
    parser.add_argument(
        "--no-pr-curve",
        action="store_true",
        help="Skip recomputing the PR curve (used by --check, which has no artefacts in CI).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    include_pr_curve = not args.no_pr_curve and not args.check
    payload = build_export(settings, include_pr_curve=include_pr_curve)
    rendered = render_typescript(payload)

    if args.check:
        if not args.output.exists():
            logger.error("%s does not exist; run without --check first", args.output)
            return 1
        existing = args.output.read_text(encoding="utf-8")
        # generated_at and the PR curve are excluded from the comparison: the timestamp
        # always differs, and --check never recomputes the curve (no artefacts in CI).
        existing_payload = _strip_volatile(_extract_payload(existing))
        current_payload = _strip_volatile(payload)
        if existing_payload != current_payload:
            logger.error(
                "%s is stale relative to %s -- re-run without --check",
                args.output,
                settings.registry_path,
            )
            return 1
        logger.info("%s matches the current registry (PR curve not checked)", args.output)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    logger.info("wrote %s", args.output)
    return 0


def _extract_payload(rendered: str) -> dict[str, Any]:
    """Pull the JSON payload back out of a previously rendered TS module.

    The JSON body is everything after ``export const METRICS: GeneratedMetrics = `` through
    the end of the file -- render_typescript emits nothing after the closing brace, on
    purpose (an ``as`` assertion there would suppress the type check this file exists for).
    """
    marker = "export const METRICS: GeneratedMetrics = "
    start = rendered.index(marker) + len(marker)
    return dict(json.loads(rendered[start:]))


def _strip_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that legitimately differ between two exports of the same registry state."""
    stripped = dict(payload)
    stripped.pop("generated_at", None)
    tier1 = stripped.get("tier1")
    if isinstance(tier1, dict):
        tier1 = dict(tier1)
        tier1.pop("pr_curve", None)
        stripped["tier1"] = tier1
    return stripped


def main() -> int:
    """Console-script entry point."""
    return run()


if __name__ == "__main__":
    sys.exit(main())
