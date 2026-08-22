"""Shared test fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Return settings pinned to the CI environment with a test-only signing key."""
    return Settings(environment="ci", jwt_secret_key="test-only-signing-key-not-a-real-secret")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Return a test client bound to a freshly built application."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
