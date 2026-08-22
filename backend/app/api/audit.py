"""``GET /audit/{transaction_id}`` — the audit trail for one decision.

Phase 7 implements this router against the append-only ``audit_log`` table. Read-only by
construction: no route here may expose an update or delete path.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/audit", tags=["audit"])
