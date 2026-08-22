"""Tests for the liveness endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    """The health endpoint reports the service as up."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "riskiq-backend"


def test_health_needs_no_credentials(client: TestClient) -> None:
    """Liveness must not require auth, or the orchestrator cannot probe it."""
    assert client.get("/health").status_code == 200


def test_health_rejects_unknown_fields_in_response_schema(client: TestClient) -> None:
    """The response carries exactly the declared fields and nothing extra."""
    body = client.get("/health").json()

    assert set(body) == {"status", "service", "version", "environment"}
