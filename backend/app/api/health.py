"""Liveness endpoint.

Intentionally unauthenticated and dependency-free: it reports that the process is up and
serving, and deliberately does not check Postgres or Redis. A liveness probe that fails
when a downstream dependency is down causes the orchestrator to restart a healthy
container. Dependency health gets its own endpoint in Phase 10.
"""

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Report that the API process is up and serving requests.

    Returns:
        A :class:`HealthResponse` with status ``"ok"``.
    """
    settings: Settings = get_settings()
    return HealthResponse(
        status="ok",
        service="riskiq-backend",
        version=settings.api_version,
        environment=settings.environment,
    )
