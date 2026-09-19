from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from janus.api.app import create_app
from janus.api.security import SecuritySettings


USER_KEY = "test-user-api-key-0000000000000001"
ADMIN_KEY = "test-admin-api-key-00000000000001"
TRUSTED_ORIGIN = "https://janus.example.test"


def secured_client(tmp_path) -> TestClient:
    settings = SecuritySettings(
        api_key=USER_KEY,
        admin_api_key=ADMIN_KEY,
        trusted_origins=frozenset({TRUSTED_ORIGIN}),
    )
    return TestClient(
        create_app(
            security_settings=settings,
            db_path=str(tmp_path / "security.db"),
        )
    )


def test_rest_endpoints_require_bearer_authentication(tmp_path):
    client = secured_client(tmp_path)
    assert client.get("/health").status_code == 401
    assert client.get("/api/sessions").status_code == 401
    assert client.get(
        "/health",
        headers={"Authorization": f"Bearer {USER_KEY}"},
    ).status_code == 200


def test_llm_configuration_requires_admin_and_forbids_base_url(tmp_path):
    client = secured_client(tmp_path)
    user_headers = {"Authorization": f"Bearer {USER_KEY}"}
    admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

    assert client.post(
        "/api/system/llm-config",
        headers=user_headers,
        json={"provider": "mock"},
    ).status_code == 403
    response = client.post(
        "/api/system/llm-config",
        headers=admin_headers,
        json={"provider": "ollama", "base_url": "https://attacker.example"},
    )
    assert response.status_code == 422


def test_cors_allows_only_configured_origin(tmp_path):
    client = secured_client(tmp_path)
    headers = {"Authorization": f"Bearer {USER_KEY}"}
    trusted = client.get("/health", headers={**headers, "Origin": TRUSTED_ORIGIN})
    attacker = client.get("/health", headers={**headers, "Origin": "https://attacker.example"})
    assert trusted.headers["access-control-allow-origin"] == TRUSTED_ORIGIN
    assert "access-control-allow-origin" not in attacker.headers


def test_websockets_require_trusted_origin_and_authentication(tmp_path):
    client = secured_client(tmp_path)
    headers = {"Authorization": f"Bearer {USER_KEY}"}
    client.post(
        "/api/sessions",
        headers=headers,
        json={
            "session_id": "secured-session",
            "speaker_a": {"speaker_id": "a", "name": "Alice", "native_language": "en"},
            "speaker_b": {"speaker_id": "b", "name": "Bob", "native_language": "es"},
        },
    )

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/live-notes/secured-session",
            headers={"Origin": "https://attacker.example"},
        ):
            pass

    with client.websocket_connect(
        "/ws/live-notes/secured-session",
        headers={"Origin": TRUSTED_ORIGIN},
    ) as websocket:
        websocket.send_json({"type": "auth", "token": USER_KEY})
        assert websocket.receive_json()["type"] == "history"


def test_http_request_bodies_are_size_limited(tmp_path):
    client = secured_client(tmp_path)
    response = client.post(
        "/api/sessions",
        headers={"Authorization": f"Bearer {USER_KEY}"},
        content=b"x" * (2 * 1024 * 1024 + 1),
    )
    assert response.status_code == 413
