"""FastAPI application entry point for the RiskIQ risk-decisioning service.

Run locally with::

    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api import audit, health, rings, score, transactions
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Configuration override, used by tests. Defaults to process settings.

    Returns:
        The configured application, with every router mounted.
    """
    cfg = settings or get_settings()
    application = FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        description="Real-time fraud, chargeback and abuse-ring detection.",
    )

    application.include_router(health.router)
    application.include_router(score.router)
    application.include_router(transactions.router)
    application.include_router(audit.router)
    application.include_router(rings.router)

    return application


app = create_app()
