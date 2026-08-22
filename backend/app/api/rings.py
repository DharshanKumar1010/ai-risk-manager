"""``GET /rings`` — Tier-3 abuse rings.

Phase 7 implements this router. Responses expose ring membership and the centrality
metrics that drove each flag, but never the decision threshold or per-feature weights —
that would make the endpoint an evasion oracle.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/rings", tags=["rings"])
