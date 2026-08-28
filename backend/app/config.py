"""Application configuration, loaded from the environment via pydantic-settings.

No default in this module may be a real credential. Secrets are supplied through the
environment or a local ``.env`` file that is never committed — see
``.claude/skills/security-checklist/SKILL.md`` section 1.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_JWT_SECRET = "dev-only-placeholder-change-me-before-deploy"

# RFC 7518 section 3.2: an HMAC key for HS256 must be at least as long as the hash
# output. PyJWT warns below this; we refuse below it.
MIN_JWT_SECRET_BYTES = 32

#: Same placeholder-refusal pattern as the JWT secret, for a key that guards a different
#: property: not authentication, but whether Tier-3's exported entity ids are reversible. See
#: ``entity_anonymization_key``'s own description.
PLACEHOLDER_ENTITY_ANONYMIZATION_KEY = "dev-only-placeholder-change-me-before-deploy"
MIN_ENTITY_ANONYMIZATION_KEY_BYTES = 32

#: Same placeholder-refusal pattern again, for the key that is the *entire* authentication
#: surface of ``POST /webhooks/razorpay/transaction`` -- that route carries no bearer token at
#: all. See ``razorpay_webhook_secret``'s own description.
PLACEHOLDER_RAZORPAY_WEBHOOK_SECRET = "dev-only-placeholder-change-me-before-deploy"
MIN_RAZORPAY_WEBHOOK_SECRET_BYTES = 32

# Repo-root ``data/`` on a host checkout (backend/app/config.py -> up three -> repo root),
# and ``/data`` inside the backend container, where ``/srv`` is the backend directory.
# docker-compose sets DATA_DIR explicitly as well, so the deployment does not depend on
# this path arithmetic being read correctly.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: Where generated reports are written. Same path arithmetic as DEFAULT_DATA_DIR, so it
#: resolves to repo-root/notebooks on a host checkout and /notebooks in the container.
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

#: Where trained weights live. Phase 2-6 reached these through
#: ``app.ml.registry.DEFAULT_ARTIFACT_DIR``, which is derived from ``__file__`` and is correct
#: on a host checkout but wrong in the container layout, where ``/srv`` is the backend
#: directory and there is no repo root above it. Serving must not depend on that arithmetic,
#: so the directory becomes configuration here and docker-compose sets it explicitly.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class Settings(BaseSettings):
    """Runtime configuration for the RiskIQ backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Literal["local", "ci", "staging", "production"] = Field(
        description="No default, deliberately -- see Phase 9.5's audit finding. A default of "
        "'local' meant an ENVIRONMENT variable left unset in a real deployment would silently "
        "boot with every dev-mode behavior active: the three placeholder-refusal guards below "
        "never run (they only check when 'staging'/'production' is what got set), and "
        "app.main.create_app mounts the unauthenticated POST /auth/demo-token token minter. "
        "Requiring this field turns a missing environment variable into a startup failure "
        "everywhere, local dev included -- both docker-compose.yml and .env.example already "
        "set it explicitly, so this changes nothing for either.",
    )
    debug: bool = False

    demo_mode: bool = Field(
        default=False,
        description="Independent opt-in that mounts POST /auth/demo-token outside local/ci. "
        "Deliberately NOT folded into the `environment in (\"local\", \"ci\")` check itself -- "
        "that check is the Phase 9.5 fix (see app/main.py and app/api/auth.py's module "
        "docstrings) and every other environment-gated behavior in this file, in particular "
        "reject_placeholder_secret_outside_local below, must keep meaning exactly what it "
        "already means regardless of this flag. This only ORs one extra condition onto the "
        "router mount in app.main.create_app. Defaults to False, same fail-closed default as "
        "everything else here. "
        "A security review of the first version of this flag found that granting the full "
        "local/ci persona set (including 'analyst') to an anonymous internet caller "
        "reassembles the explain-oracle evasion path Phase 7/8 built explain:read/rings:read "
        "scope gates to prevent -- see security-checklist item 8.3/8.4. So when this flag (and "
        "not local/ci) is the only reason the router is mounted, app.api.auth.mint_demo_token "
        "refuses persona='analyst' outright (403) and restricts persona='merchant' to the "
        "account ids listed in demo_account_ids below (403 for anything else, and refused by "
        "default since that list defaults to empty). Turn this on only for a judged demo "
        "window, and only after setting demo_account_ids to the specific seeded ids the demo "
        "needs -- never leave it on with an empty allowlist expecting that to be a no-op, and "
        "turn it back off once the window closes. It does not revoke tokens already minted; "
        "exposure ends up to DEMO_TOKEN_EXPIRY_SECONDS (30 minutes) after the flag is flipped "
        "off, not the instant it is flipped.",
    )
    demo_account_ids: tuple[str, ...] = Field(
        default=(),
        description="The only account ids app.api.auth.mint_demo_token will mint a merchant "
        "persona token for when demo_mode (and not local/ci) is the reason the router is "
        "mounted -- see demo_mode above. Empty by default, which refuses every account id: an "
        "operator must explicitly list the seeded demo accounts (python -m app.data.seed_demo "
        "prints them) they want the judged demo to use. Not consulted at all in local/ci, "
        "where every account id remains available as before -- this field exists only to bound "
        "what an anonymous caller can reach once demo_mode opens the route to the internet.",
    )

    api_title: str = "RiskIQ"
    api_version: str = "0.1.0"

    database_url: str = Field(
        default="postgresql+asyncpg://riskiq_app@localhost:5432/riskiq",
        description="Async SQLAlchemy DSN for PostgreSQL. Defaults to the least-privilege "
        "riskiq_app role with no password, so a deployment that forgets to set this fails to "
        "connect rather than quietly connecting as the table-owning superuser — which bypasses "
        "row-level security unconditionally and would disable every isolation policy while "
        "leaving them looking correct in the schema.",
    )
    migration_database_url: str | None = Field(
        default=None,
        description="DSN Alembic connects with. Separate from database_url on purpose: the "
        "application runs as riskiq_app, which by design cannot CREATE TABLE, GRANT or ALTER "
        "ROLE — so migrations need an admin DSN. Keeping them in one variable would mean the "
        "obvious fix for a failing migration is to point the *application* at a superuser, "
        "which silently disables every row-level-security policy. Falls back to database_url "
        "when unset, so a local superuser setup still works.",
    )
    pipeline_database_url: str | None = Field(
        default=None,
        description="DSN the Phase 1 pipeline writes through. Separate from database_url for "
        "the same reason migration_database_url is: riskiq_app holds SELECT only on "
        "transactions and accounts, and the bulk COPY in app.data.pipeline.write_postgres "
        "needs the read-write riskiq_pipeline role. Falls back to migration_database_url, "
        "then database_url, so a local superuser setup still works without extra config.",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo emitted SQL. Never enable outside local debugging.",
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis DSN, used for rate limiting and the live scoring feed.",
    )

    jwt_secret_key: str = Field(
        min_length=MIN_JWT_SECRET_BYTES,
        description="HMAC signing key for access tokens, at least 32 bytes per RFC 7518 "
        "section 3.2. No default -- Phase 9.5's audit finding: a default here meant this "
        "service could boot signing tokens with a well-known key published in tracked source "
        "whenever ENVIRONMENT was left unset. Always required now, local dev included; "
        "PLACEHOLDER_JWT_SECRET below is kept only as the value "
        "reject_placeholder_secret_outside_local refuses, for a caller that explicitly sets it.",
    )
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "riskiq"
    jwt_audience: str = "riskiq-api"
    jwt_expiry_seconds: int = Field(default=3600, ge=60, le=86_400)

    entity_anonymization_key: str = Field(
        min_length=MIN_ENTITY_ANONYMIZATION_KEY_BYTES,
        description="HMAC key behind app.models.tier3_graph.export_ring_edges's entity node "
        "ids. Unsalted SHA-256 over IEEE-CIS's shared-entity fingerprints (card1/card2/card5/"
        "addr1, device columns) is reversible by dictionary attack -- the corpus is public and "
        "each fingerprint field has a small, enumerable domain, so truncation-for-collision-"
        "resistance does nothing against a preimage search over a few hundred thousand "
        "candidates. This key is what makes the mapping non-reversible without it. Used only "
        "at training/export time (app.models.train_tier3.build_served_model) -- never written "
        "into a Tier3Model artifact, models/registry.json, or any API response. No default, "
        "same Phase 9.5 reasoning as jwt_secret_key above.",
    )

    razorpay_webhook_secret: str = Field(
        min_length=MIN_RAZORPAY_WEBHOOK_SECRET_BYTES,
        description="HMAC-SHA256 key configured in the Razorpay dashboard's webhook settings, "
        "verified against X-Razorpay-Signature by app.core.webhook_security. The *only* "
        "authentication POST /webhooks/razorpay/transaction has -- there is no bearer token on "
        "that route at all. No default, same Phase 9.5 reasoning as jwt_secret_key above.",
    )

    jwt_ws_audience: str = Field(
        default="riskiq-ws",
        description="Audience claim on a websocket ticket. Distinct from jwt_audience so a "
        "ticket cannot be presented as a bearer token against any REST route -- "
        "decode_access_token pins the audience it checks, and a ticket minted under this one "
        "fails verification everywhere except the websocket route that accepts it.",
    )
    ws_ticket_expiry_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="How long a websocket ticket is valid for. Short: the ticket exists only "
        "to get a bearer credential out of the URL query string and into the WS handshake, "
        "and a stale one sitting in a browser history entry or a proxy log should stop working "
        "quickly.",
    )

    data_dir: Path = Field(
        default=DEFAULT_DATA_DIR,
        description="Root of the dataset tree. Contents are gitignored; see data/README.md.",
    )
    reports_dir: Path = Field(
        default=DEFAULT_REPORTS_DIR,
        description="Where the pipeline writes the data-quality report.",
    )
    models_dir: Path = Field(
        default=DEFAULT_MODELS_DIR,
        description="Root of the model tree: registry.json plus the artifacts/ directory.",
    )

    # --- Serving ---------------------------------------------------------------------
    scoring_source_dataset: Literal["ieee_cis", "paysim"] = Field(
        default="ieee_cis",
        description="Which corpus's models the scoring endpoint serves. Only ieee_cis has a "
        "full four-layer stack; PaySim's Tier-1 is a simulator artefact and its Tier-3 "
        "abstains on every test transaction, so it is not a servable default.",
    )
    tier3_timeout_ms: int = Field(
        default=50,
        ge=1,
        le=5_000,
        description="Budget for the Tier-3 ring lookup. On expiry the decision is made "
        "without Tier-3 and the audit row records degraded mode and why — the Phase 7 "
        "graceful-degradation requirement, and security-checklist item 5.3.",
    )
    account_history_limit: int = Field(
        default=500,
        ge=1,
        le=10_000,
        description="Most recent prior transactions read per account when assembling "
        "serving features. Bounds the indexed range scan so one very active account cannot "
        "make a scoring call unboundedly slow. The widest engineered window is 7 days.",
    )

    # --- Rate limiting ---------------------------------------------------------------
    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Requests permitted per principal per window on public endpoints.",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3_600,
        description="Length of the fixed rate-limit window.",
    )

    cors_allow_origins: tuple[str, ...] = Field(
        default=("http://localhost:5173",),
        description="Exact origins the dashboard is served from. Never '*': these endpoints "
        "are credentialed, and a wildcard with credentials is both refused by browsers and "
        "wrong to ask for.",
    )

    @property
    def raw_data_dir(self) -> Path:
        """Directory holding untouched dataset downloads (IEEE-CIS, PaySim)."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory holding pipeline output — the parquet materialisations."""
        return self.data_dir / "processed"

    @property
    def alembic_url(self) -> str:
        """Return the DSN migrations should run against."""
        return self.migration_database_url or self.database_url

    @property
    def pipeline_url(self) -> str:
        """Return the DSN the Phase 1 pipeline and the demo seed script write through."""
        return self.pipeline_database_url or self.migration_database_url or self.database_url

    @property
    def artifact_dir(self) -> Path:
        """Directory holding trained weights. Gitignored; the registry is the tracked part."""
        return self.models_dir / "artifacts"

    @property
    def registry_path(self) -> Path:
        """The append-only ``registry.json`` that resolves a model_version to its provenance."""
        return self.models_dir / "registry.json"

    @model_validator(mode="after")
    def reject_wildcard_cors_origin(self) -> "Settings":
        """Refuse a wildcard origin.

        Every route on this service except ``/health`` is credentialed, and a wildcard origin
        cannot be combined with credentials — browsers reject the pair outright. Catching it
        here turns a confusing runtime CORS failure into a startup error that names the cause.
        """
        if "*" in self.cors_allow_origins:
            raise ValueError(
                "cors_allow_origins must list exact origins; '*' cannot be used on a "
                "credentialed API. Set the dashboard's origin explicitly."
            )
        return self

    @model_validator(mode="after")
    def reject_ws_audience_equal_to_api_audience(self) -> "Settings":
        """Refuse a deployment where the websocket ticket audience equals the REST audience.

        The entire reason ``mint_ws_ticket`` mints a *separate*-audience token is that a
        websocket ticket travels somewhere a normal bearer token must not: the URL query
        string, and therefore uvicorn's access log, any reverse proxy's log, and browser
        history (see ``app/core/security.py``'s docstring). If ``JWT_WS_AUDIENCE`` were ever
        set equal to ``JWT_AUDIENCE`` -- both are independently env-settable -- every ticket
        would decode successfully against every REST route too, and a full-privilege bearer
        token would be riding in a URL on every live-feed connection. Caught at startup rather
        than left to be discovered the day a log line turns out to be a valid credential.
        """
        if self.jwt_ws_audience == self.jwt_audience:
            raise ValueError(
                "jwt_ws_audience must differ from jwt_audience -- see mint_ws_ticket's "
                "docstring in app/core/security.py for why a shared audience defeats the "
                "point of a separate websocket ticket."
            )
        return self

    @model_validator(mode="after")
    def reject_placeholder_secret_outside_local(self) -> "Settings":
        """Refuse to boot a deployed environment while still using a placeholder secret.

        Failing at startup is deliberate: a service that silently runs on a well-known
        signing key -- or a well-known entity-anonymization key, which makes every exported
        Tier-3 entity id reversible by anyone who has read this file on GitHub -- is worse than
        one that does not start.
        """
        deployed = self.environment in ("staging", "production")
        if deployed and self.jwt_secret_key == PLACEHOLDER_JWT_SECRET:
            raise ValueError(
                f"jwt_secret_key is still the placeholder in environment={self.environment!r}. "
                "Set JWT_SECRET_KEY to a real value."
            )
        if deployed and self.entity_anonymization_key == PLACEHOLDER_ENTITY_ANONYMIZATION_KEY:
            raise ValueError(
                f"entity_anonymization_key is still the placeholder in "
                f"environment={self.environment!r}. Set ENTITY_ANONYMIZATION_KEY to a real "
                "value before training/exporting Tier-3."
            )
        if deployed and self.razorpay_webhook_secret == PLACEHOLDER_RAZORPAY_WEBHOOK_SECRET:
            raise ValueError(
                f"razorpay_webhook_secret is still the placeholder in "
                f"environment={self.environment!r}. Set RAZORPAY_WEBHOOK_SECRET to the value "
                "configured in the Razorpay dashboard's webhook settings."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is read once per process and so that FastAPI can use
    this directly as a dependency without re-parsing the environment on every request.
    """
    return Settings()
