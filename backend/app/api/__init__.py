"""HTTP routers.

``health`` is live from Phase 0. The remaining routers are declared here so the
application's URL surface is visible from the start; their routes are added in Phase 7
(``score``, ``transactions``, ``audit``, ``rings``) behind the JWT dependency in
``app.core.security``.
"""

from app.api import audit, health, rings, score, transactions

__all__ = ["audit", "health", "rings", "score", "transactions"]
