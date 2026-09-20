import json
import pytest
from unittest.mock import MagicMock
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    LiveMeetingNotes,
    ActionItem,
)
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.llm.mock_llm_adapter import MockLLMAdapter
from janus.services.live_notetaker_service import LiveNotetakerService


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_live_notes.db")
    return SqliteMeetingRepository(db_path=db_file)


@pytest.fixture
def broadcaster():
    b = MagicMock()
    b.broadcast = MagicMock()
    return b


def test_live_notetaker_triggers_on_batch_size(repo, broadcaster):
    # Prepare meeting
    m = Meeting(
        meeting_id="meet_live_1",
        title="Discusión de Producto",
        speaker_a=SpeakerProfile("carlos", "Carlos", "es"),
        speaker_b=SpeakerProfile("alice", "Alice", "en"),
    )
    repo.save_meeting(m)

    # Mock LLM returning JSON with structured notes
    llm_json_response = json.dumps({
        "current_topic": "Definición de Arquitectura S2ST",
        "key_takeaways": [
            "Se definió utilizar SQLite FTS5 para memoria histórica",
            "Se integrará Google Gemini y Ollama como opciones BYOM"
        ],
        "action_items": [
            {"assignee": "Carlos", "task": "Configurar FTS5 con tokenizador unicode61", "due_hint": "Hoy"}
        ]
    })
    llm = MockLLMAdapter(default_response=llm_json_response)

    service = LiveNotetakerService(
        repository=repo,
        llm_provider=llm,
        broadcaster=broadcaster,
        batch_size=2,  # Trigger every 2 turns for testing
    )

    # 1st turn
    turn1 = ConversationTurn(
        turn_id="t1",
        session_id="meet_live_1",
        speaker_id="carlos",
        original_transcription=TranscriptionResult("Hablemos de la arquitectura del proyecto", "es"),
        translation=TranslationResult("Hablemos de la arquitectura del proyecto", "es", "Let's talk about the project architecture", "en"),
    )
    repo.save_turn("meet_live_1", turn1)
    notes1 = service.process_turn("meet_live_1", turn1)
    assert notes1 is None  # Not yet triggered (batch_size=2)
    assert broadcaster.broadcast.call_count == 0

    # 2nd turn
    turn2 = ConversationTurn(
        turn_id="t2",
        session_id="meet_live_1",
        speaker_id="alice",
        original_transcription=TranscriptionResult("I will configure FTS5 and Carlos will work on BYOM", "en"),
        translation=TranslationResult("I will configure FTS5 and Carlos will work on BYOM", "en", "Configuraré FTS5 y Carlos trabajará en BYOM", "es"),
    )
    repo.save_turn("meet_live_1", turn2)
    notes2 = service.process_turn("meet_live_1", turn2)

    assert notes2 is not None
    assert notes2.current_topic == "Definición de Arquitectura S2ST"
    assert len(notes2.key_takeaways) == 2
    assert "SQLite FTS5" in notes2.key_takeaways[0]
    assert len(notes2.action_items) == 1
    assert notes2.action_items[0].assignee == "Carlos"

    # Broadcaster should have been called with broadcast_event
    assert broadcaster.broadcast_event.call_count >= 1
    event = broadcaster.broadcast_event.call_args[0][1]
    assert event.event_name == "LiveNotesUpdated"
    assert event.current_topic == "Definición de Arquitectura S2ST"


def test_live_notetaker_catch_up(repo):
    m = Meeting(
        meeting_id="meet_catch_up",
        title="Sincronización Semanal",
        speaker_a=SpeakerProfile("carlos", "Carlos", "es"),
        speaker_b=SpeakerProfile("alice", "Alice", "en"),
    )
    repo.save_meeting(m)

    for i in range(4):
        t = ConversationTurn(
            turn_id=f"t_{i}",
            session_id="meet_catch_up",
            speaker_id="carlos" if i % 2 == 0 else "alice",
            original_transcription=TranscriptionResult(f"Actualización del punto {i}", "es"),
            translation=TranslationResult(f"Actualización del punto {i}", "es", f"Update on point {i}", "en"),
        )
        repo.save_turn("meet_catch_up", t)

    llm = MockLLMAdapter(default_response="En los últimos minutos, Carlos y Alice acordaron las prioridades del sprint.")
    service = LiveNotetakerService(repository=repo, llm_provider=llm)

    recap = service.catch_up("meet_catch_up", last_n_turns=3)
    assert "Carlos y Alice" in recap
