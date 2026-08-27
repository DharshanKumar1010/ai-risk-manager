"""HTTP routers.

``health`` is live from Phase 0. ``score``, ``transactions``, ``audit`` and ``rings`` were
added in Phase 7, behind the JWT dependency in ``app.core.security``. ``auth`` (the
demo-token walkthrough endpoint) and ``feed`` (the live scoring feed's ticket and WebSocket
routes) were added in Phase 8. ``auth`` is mounted conditionally -- see its module docstring
and :func:`app.main.create_app`; ``feed`` is mounted unconditionally, since minting a ticket
and opening the socket are real product features, not a walkthrough convenience.
"""

from app.api import audit, auth, feed, health, rings, score, transactions

__all__ = ["audit", "auth", "feed", "health", "rings", "score", "transactions"]
