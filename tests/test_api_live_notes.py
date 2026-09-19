import json
from fastapi.testclient import TestClient
from janus.api.app import create_app
from janus.services.session_service import SessionService
from janus.services.meeting_notes_service import MeetingNotesService
from janus.services.live_notetaker_service import LiveNotetakerService
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.adapters.llm.mock_llm_adapter import MockLLMAdapter
from janus.domain.models import ConversationTurn, TranscriptionResult, TranslationResult


AUTH_HEADERS = {"Authorization": "Bearer test-user-api-key-0000000000000001"}


def test_api_live_notes_and_catch_up(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "live_notes_api_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    mock_llm_response = json.dumps({
        "current_topic": "Acuerdos de Integración",
        "key_takeaways": ["Se acordó probar con Gemini 3.6 Flash"],
        "action_items": [{"assignee": "Carlos", "task": "Preparar demo", "due_hint": "Pronto"}]
    })
    mock_llm = MockLLMAdapter(default_response=mock_llm_response)
    live_notetaker = LiveNotetakerService(
        repository=meeting_repo,
        llm_provider=mock_llm,
        broadcaster=broadcaster,
        batch_size=2,
    )

    app = create_app(
        session_service=session_service,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
        notes_service=notes_service,
        live_notetaker=live_notetaker,
        llm_provider=mock_llm,
    )
    client = TestClient(app, headers=AUTH_HEADERS)

    # 1. Create meeting
    meet_payload = {
        "meeting_id": "meet_zoom_companion",
        "title": "Sincronización de Producto",
        "speaker_a": {"speaker_id": "carlos", "name": "Carlos", "native_language": "es"},
        "speaker_b": {"speaker_id": "alice", "name": "Alice", "native_language": "en"},
    }
    res = client.post("/api/meetings", json=meet_payload)
    assert res.status_code == 201

    # 2. Query initial live notes (should have default topic)
    res_notes = client.get("/api/meetings/meet_zoom_companion/live-notes")
    assert res_notes.status_code == 200
    data = res_notes.json()
    assert data["meeting_id"] == "meet_zoom_companion"
    assert "current_topic" in data

    # 3. Add turn and force refresh
    t1 = ConversationTurn(
        turn_id="t1",
        session_id="meet_zoom_companion",
        speaker_id="carlos",
        original_transcription=TranscriptionResult("Acordamos probar con Gemini 3.6 Flash", "es"),
        translation=TranslationResult("Acordamos probar con Gemini 3.6 Flash", "es", "We agreed to test with Gemini 3.6 Flash", "en"),
    )
    meeting_repo.save_turn("meet_zoom_companion", t1)

    res_refresh = client.post("/api/meetings/meet_zoom_companion/live-notes/refresh")
    assert res_refresh.status_code == 200
    refreshed_data = res_refresh.json()
    assert refreshed_data["current_topic"] == "Acuerdos de Integración"
    assert len(refreshed_data["key_takeaways"]) >= 1

    # 4. Catch up
    res_catch_up = client.post("/api/meetings/meet_zoom_companion/catch-up", json={"last_n_turns": 3})
    assert res_catch_up.status_code == 200
    catch_up_data = res_catch_up.json()
    assert catch_up_data["meeting_id"] == "meet_zoom_companion"
    assert "summary" in catch_up_data
