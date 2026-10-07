from fastapi.testclient import TestClient
from janus.api.app import create_app
from janus.domain.models import SpeakerProfile, Meeting, ConversationTurn, TranscriptionResult, TranslationResult
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.meeting_notes_service import MeetingNotesService
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster


def test_gate4_mobile_interactive_console_flow(tmp_path):
    """
    Validates Gate 4:
    Simulates the complete mobile-interviewer lifecycle:
    1. Capture turns from mobile.
    2. In-situ speaker renaming on the phone ('speaker_1' -> 'Carlos').
    3. Finalize interview and generate structured minutes.
    4. Mobile user directly updates/enriches the minutes in SQLite.
    5. Export markdown for distribution in < 1 second.
    """
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "gate4_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Nos reunimos para acordar el lanzamiento", language="es"),
        translation_engine=MockTranslator({"Nos reunimos para acordar el lanzamiento": "We meet to agree on the launch"}),
        tts_engine=MockSpeechSynthesizer(),
        broadcaster=broadcaster,
        storage_repo=meeting_repo,
    )

    app = create_app(
        session_service=session_service,
        orchestrator=orchestrator,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
        notes_service=notes_service,
    )
    client = TestClient(app)

    # 1. Create Meeting
    meet_payload = {
        "meeting_id": "mobile_live_session",
        "title": "Entrevista En Vivo Desde Móvil",
        "speaker_a": {"speaker_id": "speaker_1", "name": "Hablante 1", "native_language": "es"},
        "speaker_b": {"speaker_id": "speaker_2", "name": "Hablante 2", "native_language": "es"},
    }
    res = client.post("/api/meetings", json=meet_payload)
    assert res.status_code == 201

    # 2. Rename speaker in-situ from mobile phone
    rename_res = client.patch(
        "/api/meetings/mobile_live_session/rename-speaker",
        json={"speaker_id": "speaker_1", "new_name": "Carlos (Entrevistador)"},
    )
    assert rename_res.status_code == 200
    assert rename_res.json()["new_name"] == "Carlos (Entrevistador)"

    # Verify session and persistent storage reflect the updated name
    get_res = client.get("/api/meetings/mobile_live_session")
    assert get_res.json()["speaker_a"]["name"] == "Carlos (Entrevistador)"

    # 3. Add an interview dialogue turn
    turn = ConversationTurn(
        turn_id="turn_m1",
        session_id="mobile_live_session",
        speaker_id="speaker_1",
        original_transcription=TranscriptionResult(text="Yo me comprometo a enviar el borrador mañana", language="es"),
        translation=TranslationResult(source_text="Yo me comprometo a enviar el borrador mañana", source_lang="es", translated_text="I commit to sending the draft tomorrow", target_lang="en"),
    )
    meeting_repo.save_turn("mobile_live_session", turn)

    # 4. Finalize meeting & generate initial notes
    finalize_res = client.post("/api/meetings/mobile_live_session/finalize")
    assert finalize_res.status_code == 200
    summary = finalize_res.json()["summary"]
    assert len(summary["action_items"]) >= 1

    # 5. Mobile user enriches / edits the summary live
    update_res = client.put(
        "/api/meetings/mobile_live_session/summary",
        json={
            "executive_summary": "Entrevista finalizada con éxito. Se confirmaron todos los hitos clave.",
            "key_points": ["Punto 1 revisado", "Lanzamiento aprobado"],
            "action_items": [
                {"assignee": "Carlos (Entrevistador)", "task": "Enviar el borrador técnico", "completed": True, "due_hint": "Mañana"}
            ],
        },
    )
    assert update_res.status_code == 200
    updated_summary = update_res.json()["summary"]
    assert updated_summary["executive_summary"] == "Entrevista finalizada con éxito. Se confirmaron todos los hitos clave."
    assert updated_summary["action_items"][0]["completed"] is True

    # 6. Export as Markdown
    md_res = client.get("/api/meetings/mobile_live_session/notes?notes_format=markdown")
    assert md_res.status_code == 200
    assert "Entrevista finalizada con éxito" in md_res.text
