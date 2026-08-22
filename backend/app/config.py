"""Application configuration, loaded from the environment via pydantic-settings.

No default in this module may be a real credential. Secrets are supplied through the
environment or a local ``.env`` file that is never committed — see
``.claude/skills/security-checklist/SKILL.md`` section 1.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_JWT_SECRET = "dev-only-placeholder-change-me-before-deploy"

# RFC 7518 section 3.2: an HMAC key for HS256 must be at least as long as the hash
# output. PyJWT warns below this; we refuse below it.
MIN_JWT_SECRET_BYTES = 32


class Settings(BaseSettings):
    """Runtime configuration for the RiskIQ backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Literal["local", "ci", "staging", "production"] = "local"
    debug: bool = False

    api_title: str = "RiskIQ"
    api_version: str = "0.1.0"

    database_url: str = Field(
        default="postgresql+asyncpg://riskiq:riskiq@localhost:5432/riskiq",
        description="Async SQLAlchemy DSN for PostgreSQL.",
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
        default=PLACEHOLDER_JWT_SECRET,
        min_length=MIN_JWT_SECRET_BYTES,
        description="HMAC signing key for access tokens. Must be overridden outside local, "
        "and at least 32 bytes per RFC 7518 section 3.2.",
    )
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "riskiq"
    jwt_audience: str = "riskiq-api"
    jwt_expiry_seconds: int = Field(default=3600, ge=60, le=86_400)

    @model_validator(mode="after")
    def reject_placeholder_secret_outside_local(self) -> "Settings":
        """Refuse to boot a deployed environment while still using the placeholder secret.

        Failing at startup is deliberate: a service that silently runs on a well-known
        signing key is worse than one that does not start.
        """
        deployed = self.environment in ("staging", "production")
        if deployed and self.jwt_secret_key == PLACEHOLDER_JWT_SECRET:
            raise ValueError(
                f"jwt_secret_key is still the placeholder in environment={self.environment!r}. "
                "Set JWT_SECRET_KEY to a real value."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is read once per process and so that FastAPI can use
    this directly as a dependency without re-parsing the environment on every request.
    """
    return Settings()
