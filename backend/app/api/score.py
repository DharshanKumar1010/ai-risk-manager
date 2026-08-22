"""``POST /score`` — run a transaction through all four layers.

Phase 7 implements this router. Every route added here requires the JWT dependency from
``app.core.security``, writes through ``app.core.audit.write_audit_record``, and applies
the Tier-3 timeout with the degraded-mode fallback.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/score", tags=["scoring"])
