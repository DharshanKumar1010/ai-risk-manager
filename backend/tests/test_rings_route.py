"""``GET /rings`` -- response shape, including the Phase 8 ``nodes``/``edges`` topology."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.security import SCOPE_ANALYST, SCOPE_RINGS_READ
from app.models.tier3_graph import RingGraphEdge, RingGraphNode, Tier3Model
from tests.conftest import auth_header


def _ring_scoped_headers(settings: Settings) -> dict[str, str]:
    return auth_header(
        settings, subject="analyst-1", account_id=None, scopes=(SCOPE_RINGS_READ, SCOPE_ANALYST)
    )


def _tier3_with_one_flagged_ring() -> Tier3Model:
    """A minimal but real Tier3Model: one ring above threshold, with topology attached."""
    return Tier3Model(
        model_id="tier3-graph-louvain-ieee-cis-test",
        source_dataset="ieee_cis",
        threshold=0.5,
        scores={"a0": 0.9, "a1": 0.9, "a2": 0.9},
        ring_of={"a0": "r0", "a1": "r0", "a2": "r0"},
        ring_sizes={"r0": 3},
        snapshot_end=datetime(2026, 1, 1, tzinfo=UTC),
        ring_nodes={
            "r0": (
                RingGraphNode(node_id="a0", kind="account"),
                RingGraphNode(node_id="a1", kind="account"),
                RingGraphNode(node_id="a2", kind="account"),
                RingGraphNode(node_id="9f8e7d6c5b4a3210", kind="entity", entity_type="device_fp"),
            )
        },
        ring_edges={
            "r0": (
                RingGraphEdge(source="a0", target="9f8e7d6c5b4a3210"),
                RingGraphEdge(source="a1", target="9f8e7d6c5b4a3210"),
                RingGraphEdge(source="a2", target="9f8e7d6c5b4a3210"),
            )
        },
    )


def test_a_flagged_ring_carries_its_topology(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    app.state.model_bundle = SimpleNamespace(tier3=_tier3_with_one_flagged_ring())

    response = client.get("/rings", headers=_ring_scoped_headers(settings))

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    ring = body["rings"][0]
    assert ring["ring_id"] == "r0"
    assert {member["account_id"] for member in ring["members"]} == {"a0", "a1", "a2"}

    node_ids = {node["node_id"] for node in ring["nodes"]}
    assert node_ids == {"a0", "a1", "a2", "9f8e7d6c5b4a3210"}
    entity_nodes = [node for node in ring["nodes"] if node["kind"] == "entity"]
    assert entity_nodes and entity_nodes[0]["entity_type"] == "device_fp"
    assert len(ring["edges"]) == 3
    for edge in ring["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_a_ring_below_threshold_is_not_reported_at_all(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    """The existing membership filter -- unaffected by adding topology -- still applies."""
    model = _tier3_with_one_flagged_ring()
    below_threshold = replace(model, threshold=0.99)
    app.state.model_bundle = SimpleNamespace(tier3=below_threshold)

    response = client.get("/rings", headers=_ring_scoped_headers(settings))

    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_a_ring_trained_before_phase_8_reports_empty_topology(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    """An older artifact with no stored `ring_nodes`/`ring_edges` must not 500 the route."""
    model = _tier3_with_one_flagged_ring()
    untopologized = replace(model, ring_nodes={}, ring_edges={})
    app.state.model_bundle = SimpleNamespace(tier3=untopologized)

    response = client.get("/rings", headers=_ring_scoped_headers(settings))

    assert response.status_code == 200
    ring = response.json()["rings"][0]
    assert ring["nodes"] == []
    assert ring["edges"] == []


def test_no_snapshot_loaded_still_returns_503(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    app.state.model_bundle = None

    response = client.get("/rings", headers=_ring_scoped_headers(settings))

    assert response.status_code == 503


def test_nodes_and_edges_are_capped_consistently_with_members(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    """Security-review regression cover: `nodes`/`edges` used to be shipped uncapped even
    though `members` was capped at MAX_MEMBERS, so a large ring's response size was unbounded
    and disagreed with its own `members` list about who was in the ring. `export_ring_edges`
    lists every account node before any entity node, so a naive cap at MAX_MEMBERS on `nodes`
    alone would also have dropped every entity node from an oversized ring outright -- this
    pins that a survivable entity (reachable from a *kept* account) stays, and one reachable
    only from a *dropped* account does not.
    """
    from app.api.rings import MAX_MEMBERS

    accounts = [f"a{i:04d}" for i in range(MAX_MEMBERS + 5)]
    scores = {account: 0.9 for account in accounts}
    ring_of = {account: "r0" for account in accounts}
    kept_accounts = sorted(accounts)[:MAX_MEMBERS]
    dropped_account = next(a for a in accounts if a not in kept_accounts)
    kept_survivor_account = kept_accounts[0]

    nodes = tuple(RingGraphNode(node_id=a, kind="account") for a in accounts) + (
        RingGraphNode(node_id="entity-survives", kind="entity", entity_type="device_fp"),
        RingGraphNode(node_id="entity-dropped", kind="entity", entity_type="device_fp"),
    )
    edges = (
        RingGraphEdge(source=kept_survivor_account, target="entity-survives"),
        RingGraphEdge(source=dropped_account, target="entity-dropped"),
    )
    model = Tier3Model(
        model_id="tier3-graph-louvain-ieee-cis-test",
        source_dataset="ieee_cis",
        threshold=0.5,
        scores=scores,
        ring_of=ring_of,
        ring_sizes={"r0": len(accounts)},
        snapshot_end=datetime(2026, 1, 1, tzinfo=UTC),
        ring_nodes={"r0": nodes},
        ring_edges={"r0": edges},
    )
    app.state.model_bundle = SimpleNamespace(tier3=model)

    response = client.get("/rings", headers=_ring_scoped_headers(settings), params={"limit": 1})

    assert response.status_code == 200
    ring = response.json()["rings"][0]
    member_ids = {m["account_id"] for m in ring["members"]}
    node_ids = {n["node_id"] for n in ring["nodes"]}

    assert len(ring["members"]) == MAX_MEMBERS
    assert dropped_account not in member_ids
    # nodes must not silently disagree with members about who is in the ring.
    assert member_ids <= node_ids
    assert dropped_account not in node_ids
    assert "entity-survives" in node_ids, "an entity reachable from a kept account must survive"
    assert (
        "entity-dropped" not in node_ids
    ), "an entity reachable only from a dropped account must not"
    edge_targets = {(e["source"], e["target"]) for e in ring["edges"]}
    assert (kept_survivor_account, "entity-survives") in edge_targets
    assert not any("entity-dropped" in pair for pair in edge_targets)
