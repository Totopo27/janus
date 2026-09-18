from fastapi.testclient import TestClient
from janus.api.app import create_app
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster


def test_api_health_and_session_routes():
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(),
        translation_engine=MockTranslator(),
        tts_engine=MockSpeechSynthesizer(),
        broadcaster=broadcaster,
    )

    app = create_app(
        session_service=session_service,
        orchestrator=orchestrator,
        broadcaster=broadcaster,
    )
    client = TestClient(app)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["app"] == "Janus"

    # 2. Create session
    payload = {
        "session_id": "api_test_sess",
        "speaker_a": {
            "speaker_id": "spk_1",
            "name": "Lucía",
            "native_language": "es",
        },
        "speaker_b": {
            "speaker_id": "spk_2",
            "name": "David",
            "native_language": "en",
        },
    }
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["session_id"] == "api_test_sess"
    assert data["speaker_a"]["name"] == "Lucía"

    # 3. List sessions
    res = client.get("/api/sessions")
    assert res.status_code == 200
    sessions = res.json()
    assert len(sessions) == 1

    # 4. Get session
    res = client.get("/api/sessions/api_test_sess")
    assert res.status_code == 200
    assert res.json()["session_id"] == "api_test_sess"

    # 5. Export transcript
    res = client.get("/api/sessions/api_test_sess/transcript")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 6. Delete session
    res = client.delete("/api/sessions/api_test_sess")
    assert res.status_code == 200
    assert res.json()["closed"] is True

    # 7. Get non-existent session
    res = client.get("/api/sessions/non_existent")
    assert res.status_code == 404
