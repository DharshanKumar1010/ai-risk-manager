"""FastAPI application entry point for the RiskIQ risk-decisioning service.

Run locally with::

    uvicorn app.main:app --reload

Three things happen here that are worth reading rather than skimming.

**Models load at startup, not per request and not at import.** Per request would put a 13 MB
booster read on the latency budget. At import would make this module unimportable wherever the
artefacts are absent, which is most of CI and every lint run. The lifespan hook is the one
place with both a running process and permission to do slow work.

**A missing model is not a failed startup.** ``POST /score`` returns 503 and ``/health`` keeps
answering, so an orchestrator does not restart-loop a container whose only problem is an
unmounted volume. The failure is logged at ``error`` with the exception type, because a service
that quietly serves 503 forever is worse than one that crashes.

**Unhandled exceptions never reach the caller.** The handler below returns a fixed body and
logs the detail server-side — security-checklist item 4.4, and the reason FastAPI's default
behaviour is not sufficient here: ``debug=True`` in a misconfigured deployment would otherwise
put a traceback, and the query that produced it, in a response body.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import audit, auth, feed, health, rings, score, transactions, webhooks
from app.config import Settings, get_settings
from app.core.feed import FeedBroadcaster
from app.core.rate_limit import build_rate_limiter
from app.core.serving import ModelBundle, shutdown_tier3_executor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load models on startup and release the limiter's connections on shutdown."""
    settings: Settings = application.state.settings
    try:
        application.state.model_bundle = ModelBundle.load(settings)
        logger.info(
            "loaded scoring models: %s",
            ", ".join(
                f"{layer}={model_id}"
                for layer, model_id in sorted(application.state.model_bundle.model_versions.items())
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        # Deliberately not fatal. See the module docstring.
        application.state.model_bundle = None
        logger.error(
            "scoring models failed to load (%s: %s); /score will return 503",
            type(exc).__name__,
            exc,
        )

    try:
        yield
    finally:
        limiter = getattr(application.state, "rate_limiter", None)
        if limiter is not None:
            await limiter.close()
        shutdown_tier3_executor()


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
        lifespan=lifespan,
    )

    # Settings live on application state so that request-scoped code — token verification in
    # particular — reads the configuration this app was built with rather than the process
    # singleton. Without it a test's signing key would be silently inert.
    application.state.settings = cfg
    application.state.rate_limiter = build_rate_limiter(cfg)
    application.state.model_bundle = None
    # In-process fan-out for the live scoring feed. See app/core/feed.py's module docstring
    # for the stated limit: this does not survive a multi-worker deployment.
    application.state.feed_broadcaster = FeedBroadcaster()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Return a fixed body for anything unhandled, and log the detail server-side."""
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    application.include_router(health.router)
    application.include_router(score.router)
    application.include_router(transactions.router)
    application.include_router(audit.router)
    application.include_router(rings.router)
    application.include_router(feed.auth_router)
    application.include_router(feed.feed_router)
    # Mounted unconditionally, unlike auth.router below -- Razorpay must be able to reach this
    # in every deployed environment, not only local/ci.
    application.include_router(webhooks.router)

    if cfg.environment in ("local", "ci") or cfg.demo_mode:
        # Registered conditionally, not gated inside the handler -- see auth.py's module
        # docstring for why. With demo_mode off and outside local/ci the path does not exist
        # at all: it 404s like any unrouted path, and never appears in /openapi.json.
        # `demo_mode` (see Settings.demo_mode) is a second, independent OR-condition for a
        # judged demo window -- it does not change what `environment in ("staging",
        # "production")` gates anywhere else in this file or in config.py.
        if cfg.environment not in ("local", "ci"):
            # Visible signal that a security-relevant mode is active outside local/ci -- see
            # Settings.demo_mode for why this must not be a silent flip.
            logger.warning(
                "demo_mode is enabled in environment=%r: POST /auth/demo-token is mounted "
                "and reachable by any caller, restricted to merchant persona and "
                "demo_account_ids=%r",
                cfg.environment,
                cfg.demo_account_ids,
            )
        application.include_router(auth.router)

    return application


app = create_app()
