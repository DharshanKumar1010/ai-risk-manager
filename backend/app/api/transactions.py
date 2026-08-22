"""``GET /transactions`` — scored transaction history.

Phase 7 implements this router. Reads are account-scoped: the account filter comes from
the verified principal, never from a client-supplied parameter.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/transactions", tags=["transactions"])
