"""Tier-3: transaction-network graph abuse-ring detection.

Phase 4 implements this layer: an account/device/card/IP graph built from PaySim's
origin-destination structure, with Louvain community detection plus degree and
betweenness centrality to surface collusion rings, exposing

    flag_rings(graph_snapshot) -> list[RingFlag]

This is the project's differentiator. Evaluation is at ring level, not node level.
"""
