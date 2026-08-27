"""``app.ml.export_metrics`` — the pure assembly and round-trip logic, no artefacts required.

``compute_pr_curve`` needs the Tier-1 booster and the processed parquet, neither of which CI
has; it is exercised by hand, the same way ``tests/test_serving.py``'s artefact-dependent tests
skip when ``models/artifacts/`` is absent. What belongs in a fast, artefact-free suite is the
registry-block selection, the render/extract round-trip, and the volatility stripping that
makes ``--check`` mode meaningful.
"""

import json

from app.ml.export_metrics import (
    _causal_cost_block,
    _extract_payload,
    _first_ope_caveat,
    _policy_summaries,
    _scrub_false_positive_cost,
    _scrub_operating_point,
    _strip_thresholds,
    _strip_volatile,
    _tier1_block,
    render_typescript,
)

#: A realistic false_positive_cost block, shaped like the real registry's -- see
#: models/registry.json's tier1_anomaly/ieee_cis entry for the fields this mirrors.
_RAW_COST_BLOCK = {
    "threshold": 0.004715553611386762,
    "threshold_criterion": "minimising estimated cost on validation",
    "flag_rate": 0.289317,
    "flagged": 25628,
    "review_cost_per_false_positive": 3.0,
    "flat_chargeback_fee_per_false_negative": 15.0,
    "false_positives": 22924,
    "false_negatives": 379,
    "false_positive_cost": 68772.0,
    "false_negative_cost": 51880.57,
    "total_cost": 120652.57,
    "units": "IEEE-CIS amount units (consistent with USD)",
    "assumptions": [
        "A false positive costs 3.00 IEEE-CIS amount units (consistent with USD) - analyst "
        "time on a manual review, or the lost margin and friction of declining a good "
        "customer.",
        "A false negative costs the transaction amount plus a flat 15.00 - the chargeback "
        "fee and internal handling on top of the value lost.",
        "Cost is linear in the number of mistakes: no queue-congestion effect, no "
        "customer-churn effect from repeated false declines, no recovery on disputed fraud.",
        "Every fraud is assumed to charge back. In practice some share is never disputed.",
        "Review capacity is unbounded.",
    ],
}

_RAW_HELDOUT = {
    "rows": 88581,
    "positives": 3083,
    "base_rate": 0.034804,
    "threshold": 0.004715553611386762,
    "threshold_criterion": "minimising estimated cost on validation",
    "pr_auc": 0.527572,
    "pr_auc_ci95": [0.5117, 0.5462],
    "pr_auc_no_skill_floor": 0.034804,
    "precision": 0.86456,
    "recall": 0.248459,
    "f1": 0.38,
    "confusion_matrix": {"tn": 62574, "fp": 22924, "fn": 379, "tp": 2704},
    "false_positive_cost": dict(_RAW_COST_BLOCK),
    "capacity_constrained_operating_point": {
        "threshold": 0.8137,
        "threshold_criterion": "flagging at most 1.0% of validation traffic",
        "false_positive_cost": dict(_RAW_COST_BLOCK),
    },
    "cost_reduction_vs_baseline": {"policy": "plug_in", "cost_delta_pct": -22.4},
    "policies": [
        {
            "strategy": "plug_in",
            "shipped_point": {
                "threshold": 488.799,
                "flag_rate": 0.010002,
                "precision": 0.82167,
                "recall": 0.236134,
                "recall_by_value": 0.268571,
                "caught_fraud_value": 126123.11,
                "missed_fraud_value": 343485.41,
                "confusion_matrix": {"tn": 1, "fp": 2, "fn": 3, "tp": 4},
                "cost_per_1000": 17.2,
            },
        }
    ],
    "cnp_regime": [
        {
            "policy": "probability",
            "threshold": 0.8517166781525443,
            "flag_rate": 0.010002,
            "precision": 0.86456,
            "recall": 0.248459,
            "recall_by_value": 0.150031,
            "caught_fraud_value": 70455.93,
            "missed_fraud_value": 399152.59,
            "confusion_matrix": {"tn": 85378, "fp": 120, "fn": 2317, "tp": 766},
            "cost_per_1000": 17652.23,
        }
    ],
    "cnp_regime_deltas": [{"policy": "plug_in", "cost_delta_pct": -2.22}],
    "sensitivity": [
        {
            "varied": "both costs",
            "factor": 1.0,
            "threshold": 0.004715553611386762,
            "total_cost": 20622.97,
            "false_positives": 100,
            "false_negatives": 5,
            "flag_rate": 0.2206,
        }
    ],
}


class TestOperatingBoundaryIsNeverExported:
    """Security-review regression cover: the live decision threshold and per-unit cost figures
    must never reach `frontend/src/data/metrics.generated.ts`, a public unauthenticated bundle.
    `app/api/schemas.py` withholds these same figures from every authenticated API response;
    this pins that the build-time export does not reopen the hole through a different channel.
    """

    def test_scrub_false_positive_cost_drops_the_threshold_and_unit_costs(self) -> None:
        scrubbed = _scrub_false_positive_cost(dict(_RAW_COST_BLOCK))
        assert scrubbed is not None
        assert "threshold" not in scrubbed
        assert "review_cost_per_false_positive" not in scrubbed
        assert "flat_chargeback_fee_per_false_negative" not in scrubbed
        # What must survive: the totals ml-evaluation-standards requires alongside a headline.
        assert scrubbed["total_cost"] == 120652.57
        assert scrubbed["false_positive_cost"] == 68772.0
        assert scrubbed["false_negative_cost"] == 51880.57
        assert scrubbed["false_positives"] == 22924
        assert scrubbed["false_negatives"] == 379

    def test_scrub_false_positive_cost_removes_assumption_sentences_stating_a_dollar_figure(
        self,
    ) -> None:
        scrubbed = _scrub_false_positive_cost(dict(_RAW_COST_BLOCK))
        assert scrubbed is not None
        joined = " ".join(scrubbed["assumptions"])
        assert "3.00" not in joined
        assert "15.00" not in joined
        # The generic methodological caveats are not collateral damage.
        assert any("linear in the number of mistakes" in text for text in scrubbed["assumptions"])
        assert any("Review capacity is unbounded" in text for text in scrubbed["assumptions"])

    def test_scrub_false_positive_cost_passes_through_none(self) -> None:
        assert _scrub_false_positive_cost(None) is None

    def test_scrub_operating_point_strips_its_own_threshold_and_nested_cost_block(self) -> None:
        scrubbed = _scrub_operating_point(
            {
                "threshold": 0.8137,
                "threshold_criterion": "flagging at most 1.0% of validation traffic",
                "false_positive_cost": dict(_RAW_COST_BLOCK),
            }
        )
        assert scrubbed is not None
        assert "threshold" not in scrubbed
        assert "threshold_criterion" not in scrubbed
        assert "threshold" not in scrubbed["false_positive_cost"]

    def test_strip_thresholds_drops_only_the_threshold_key_from_every_row(self) -> None:
        rows = [{"policy": "a", "threshold": 1.0, "flag_rate": 0.5}]
        stripped = _strip_thresholds(rows)
        assert stripped is not None
        assert "threshold" not in stripped[0]
        assert stripped[0]["policy"] == "a"
        assert stripped[0]["flag_rate"] == 0.5

    def test_tier1_block_carries_no_threshold_anywhere(self) -> None:
        entry = {"model_id": "tier1-x", "heldout_test": _RAW_HELDOUT}
        block = _tier1_block(entry)
        assert block is not None
        assert "threshold" not in block
        assert "threshold_criterion" not in block
        assert "threshold" not in block["false_positive_cost"]
        assert "threshold" not in block["capacity_constrained_operating_point"]
        assert (
            "threshold" not in block["capacity_constrained_operating_point"]["false_positive_cost"]
        )

    def test_causal_cost_block_carries_no_threshold_anywhere(self) -> None:
        entry = {
            "model_id": "causal-cost-x",
            "heldout_test": _RAW_HELDOUT,
            "notes": [
                "Break-even probability is per-transaction, not global: p > r/(A+f+r). Across "
                "the split, the median break-even sits at 0.035.",
                "A harmless methodological note that must survive.",
            ],
        }
        block = _causal_cost_block(entry)
        assert block is not None
        assert "threshold" not in block
        assert "threshold_criterion" not in block
        assert "threshold" not in block["false_positive_cost"]
        assert all("threshold" not in row for row in block["cnp_regime"])
        assert all("threshold" not in row for row in block["sensitivity"])
        assert not any(note.startswith("Break-even probability") for note in block["notes"])
        assert "A harmless methodological note that must survive." in block["notes"]

    def test_policy_summaries_carries_no_threshold(self) -> None:
        summaries = _policy_summaries(_RAW_HELDOUT["policies"])
        assert summaries is not None
        assert "threshold" not in summaries[0]
        assert summaries[0]["policy"] == "plug_in"
        assert summaries[0]["flag_rate"] == 0.010002


class TestRenderRoundTrips:
    """What --check mode depends on: render then extract must be the identity."""

    def test_a_rendered_payload_extracts_back_to_itself(self) -> None:
        payload = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "registry_sha256": "abc123",
            "tier1": {"model_id": "m1", "pr_auc": 0.5},
            "causal_cost": None,
            "meta_learner": None,
        }
        rendered = render_typescript(payload)
        assert _extract_payload(rendered) == payload

    def test_the_rendered_module_has_no_trailing_assertion(self) -> None:
        """Regression cover: an `as GeneratedMetrics` suffix would suppress the missing-field
        check the whole export design rests on -- see render_typescript's docstring.
        """
        rendered = render_typescript({"generated_at": "x", "registry_sha256": "y"})
        assert "as GeneratedMetrics" not in rendered
        assert rendered.strip().endswith("}")

    def test_the_rendered_module_is_valid_json_after_the_declaration(self) -> None:
        rendered = render_typescript({"a": 1, "b": [1, 2, 3]})
        marker = "export const METRICS: GeneratedMetrics = "
        body = rendered[rendered.index(marker) + len(marker) :]
        assert json.loads(body) == {"a": 1, "b": [1, 2, 3]}


class TestStripVolatile:
    """What --check mode is allowed to ignore, and only that."""

    def test_generated_at_is_stripped(self) -> None:
        payload = {"generated_at": "2026-01-01T00:00:00+00:00", "tier1": {"pr_auc": 0.5}}
        stripped = _strip_volatile(payload)
        assert "generated_at" not in stripped

    def test_the_pr_curve_is_stripped_but_the_rest_of_tier1_is_not(self) -> None:
        payload = {
            "tier1": {"pr_auc": 0.5, "pr_curve": {"points": [{"precision": 1.0}]}},
        }
        stripped = _strip_volatile(payload)
        assert "pr_curve" not in stripped["tier1"]
        assert stripped["tier1"]["pr_auc"] == 0.5

    def test_a_payload_with_no_tier1_block_is_unaffected(self) -> None:
        payload = {"generated_at": "x", "tier1": None}
        stripped = _strip_volatile(payload)
        assert stripped == {"tier1": None}

    def test_two_exports_differing_only_in_timestamp_and_pr_curve_compare_equal(self) -> None:
        """The property --check actually relies on."""
        first = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "registry_sha256": "abc",
            "tier1": {"pr_auc": 0.5, "pr_curve": {"points": [{"precision": 1.0}]}},
        }
        second = {
            "generated_at": "2026-06-01T12:00:00+00:00",
            "registry_sha256": "abc",
            "tier1": {"pr_auc": 0.5, "pr_curve": {"points": [{"precision": 0.9}]}},
        }
        assert _strip_volatile(first) == _strip_volatile(second)

    def test_a_genuine_metric_change_still_compares_unequal(self) -> None:
        """--check must not go blind to real drift while ignoring the volatile fields."""
        first = {"generated_at": "x", "tier1": {"pr_auc": 0.5}}
        second = {"generated_at": "y", "tier1": {"pr_auc": 0.6}}
        assert _strip_volatile(first) != _strip_volatile(second)


class TestFirstOpeCaveat:
    """ope_validation is a list of per-policy rows, not a single object -- see the finding
    that shaped this helper: the first version of this script assumed the latter and crashed.
    """

    def test_returns_the_first_rows_caveat(self) -> None:
        rows = [{"policy": "a", "caveat": "simulated"}, {"policy": "b", "caveat": "simulated"}]
        assert _first_ope_caveat(rows) == "simulated"

    def test_returns_none_for_an_empty_list(self) -> None:
        assert _first_ope_caveat([]) is None

    def test_returns_none_when_the_field_is_absent_entirely(self) -> None:
        assert _first_ope_caveat(None) is None

    def test_returns_none_for_a_non_list_value_rather_than_raising(self) -> None:
        """Regression cover for the exact bug: this used to be treated as a dict with .get()."""
        assert _first_ope_caveat({"caveat": "not actually a list"}) is None
