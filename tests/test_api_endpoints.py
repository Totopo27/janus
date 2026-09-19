from fastapi.testclient import TestClient
from janus.api.app import create_app
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.meeting_notes_service import MeetingNotesService
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster


def test_api_health_and_session_routes(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "api_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(),
        translation_engine=MockTranslator(),
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


def test_api_meeting_lifecycle_and_notes(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "meeting_api_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Yo me comprometo a preparar la demo", language="es"),
        translation_engine=MockTranslator({"Yo me comprometo a preparar la demo": "I commit to preparing the demo"}),
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

    # 1. Create Meeting via POST /api/meetings
    meet_payload = {
        "meeting_id": "board_sync_2026",
        "title": "Reunión de Directorio Janus",
        "speaker_a": {
            "speaker_id": "carlos",
            "name": "Carlos",
            "native_language": "es",
        },
        "speaker_b": {
            "speaker_id": "alice",
            "name": "Alice",
            "native_language": "en",
        },
    }
    res = client.post("/api/meetings", json=meet_payload)
    assert res.status_code == 201
    assert res.json()["title"] == "Reunión de Directorio Janus"
    assert res.json()["status"] == "active"

    # 2. List meetings
    res = client.get("/api/meetings")
    assert res.status_code == 200
    assert len(res.json()) >= 1

    # 3. Finalize meeting & generate Zoom-style notes
    res = client.post("/api/meetings/board_sync_2026/finalize")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["summary"] is not None
    assert "executive_summary" in data["summary"]

    # 4. Get notes in JSON format
    res = client.get("/api/meetings/board_sync_2026/notes")
    assert res.status_code == 200
    notes = res.json()
    assert "executive_summary" in notes

    # 5. Get notes in Markdown format
    res = client.get("/api/meetings/board_sync_2026/notes?format=markdown")
    assert res.status_code == 200
    assert "text/markdown" in res.headers["content-type"]
    assert "# Minuta de Reunión: Reunión de Directorio Janus" in res.text


def test_api_search_turns_and_topic_key(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "search_api_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    app = create_app(
        session_service=session_service,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
        notes_service=notes_service,
    )
    client = TestClient(app)

    # 1. Create meeting with topic_key
    meet_payload = {
        "meeting_id": "meet_search_test",
        "title": "Arquitectura S2ST",
        "topic_key": "proyectos/janus",
        "speaker_a": {"speaker_id": "spk_1", "name": "Carlos", "native_language": "es"},
        "speaker_b": {"speaker_id": "spk_2", "name": "Alice", "native_language": "en"},
    }
    res = client.post("/api/meetings", json=meet_payload)
    assert res.status_code == 201
    assert res.json()["topic_key"] == "proyectos/janus"

    # 2. Add turn directly via repo
    from janus.domain.models import ConversationTurn, TranscriptionResult, TranslationResult
    turn = ConversationTurn(
        turn_id="turn_s1",
        session_id="meet_search_test",
        speaker_id="spk_1",
        original_transcription=TranscriptionResult("Optimizamos el motor de inferencia local", "es"),
        translation=TranslationResult("Optimizamos el motor de inferencia local", "es", "We optimized local inference engine", "en"),
    )
    meeting_repo.save_turn("meet_search_test", turn)

    # 3. Search via GET /api/meetings/search
    res_search = client.get("/api/meetings/search?q=inferencia&topic=proyectos")
    assert res_search.status_code == 200
    results = res_search.json()
    assert len(results) == 1
    assert results[0]["turn_id"] == "turn_s1"
    assert "inferencia" in results[0]["snippet"].lower()


def test_api_meeting_chat_and_system_llm_config(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "chat_api_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)
    notes_service = MeetingNotesService(repository=meeting_repo)

    app = create_app(
        session_service=session_service,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
        notes_service=notes_service,
    )
    client = TestClient(app)

    # Create meeting
    meet_payload = {
        "meeting_id": "meet_ai_chat",
        "title": "Reunión de IA",
        "speaker_a": {"speaker_id": "spk_1", "name": "Carlos", "native_language": "es"},
        "speaker_b": {"speaker_id": "spk_2", "name": "Alice", "native_language": "en"},
    }
    client.post("/api/meetings", json=meet_payload)

    # Configure mock LLM provider
    res_cfg = client.post("/api/system/llm-config", json={"provider": "mock"})
    assert res_cfg.status_code == 200
    assert res_cfg.json()["provider"] == "mock"

    # Check config
    res_get_cfg = client.get("/api/system/llm-config")
    assert res_get_cfg.status_code == 200
    assert res_get_cfg.json()["provider"] == "mock"

    # Chat with meeting
    chat_payload = {"question": "¿De qué se habló en esta sesión?"}
    res_chat = client.post("/api/meetings/meet_ai_chat/chat", json=chat_payload)
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["meeting_id"] == "meet_ai_chat"
    assert "answer" in chat_data
    assert chat_data["provider"] == "mock"


def test_websocket_audio_stream_logging_and_turn_dispatch(tmp_path):
    import base64
    import json

    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "ws_audio_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Hola probando entrada de micrófono", language="es"),
        translation_engine=MockTranslator({"Hola probando entrada de micrófono": "Hello testing microphone input"}),
        tts_engine=MockSpeechSynthesizer(),
        broadcaster=broadcaster,
        storage_repo=meeting_repo,
    )

    app = create_app(
        session_service=session_service,
        orchestrator=orchestrator,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
    )

    client = TestClient(app)

    # 1. Create session
    sess_payload = {
        "session_id": "ws_audio_sess",
        "speaker_a": {"speaker_id": "spk_1", "name": "Carlos", "native_language": "es"},
        "speaker_b": {"speaker_id": "spk_2", "name": "Alice", "native_language": "en"},
    }
    client.post("/api/sessions", json=sess_payload)

    # 2. Connect to websocket audio stream and send audio packet
    fake_audio_bytes = b"RIFF" + b"\x00" * 300
    b64_audio = base64.b64encode(fake_audio_bytes).decode("utf-8")

    with client.websocket_connect("/ws/audio-stream/ws_audio_sess/spk_1") as websocket:
        websocket.send_text(json.dumps({"audio_base64": b64_audio}))
        response_text = websocket.receive_text()
        data = json.loads(response_text)

        assert data["type"] == "turn_result"
        assert data["original_text"] == "Hola probando entrada de micrófono"
        assert data["translated_text"] == "Hello testing microphone input"
        assert "audio_base64" in data


