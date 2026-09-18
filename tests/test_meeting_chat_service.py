import pytest
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    MeetingSummary,
    ActionItem,
    ChatMessage,
)
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.llm.mock_llm_adapter import MockLLMAdapter
from janus.services.meeting_chat_service import MeetingChatService


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_chat_service.db")
    return SqliteMeetingRepository(db_path=db_file)


def test_meeting_chat_service_answers_question(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")

    meeting = Meeting(
        meeting_id="meet_chat_1",
        title="Diseño de Arquitectura",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    repo.save_meeting(meeting)

    turn1 = ConversationTurn(
        turn_id="t_1",
        session_id="meet_chat_1",
        speaker_id="spk_carlos",
        original_transcription=TranscriptionResult("Vamos a usar SQLite FTS5 para la memoria.", "es"),
        translation=TranslationResult(
            "Vamos a usar SQLite FTS5 para la memoria.",
            "es",
            "We are going to use SQLite FTS5 for memory.",
            "en",
        ),
    )
    repo.save_turn("meet_chat_1", turn1)

    summary = MeetingSummary(
        executive_summary="Se decidió adoptar SQLite FTS5 y desacoplar puertos.",
        key_points=["SQLite FTS5 para búsqueda léxica", "BYOM para modelos"],
        action_items=[ActionItem(assignee="Carlos", task="Configurar FTS5", completed=True)],
    )
    repo.save_summary("meet_chat_1", summary)

    mock_llm = MockLLMAdapter(default_response="Se acordó usar SQLite FTS5.")
    service = MeetingChatService(repository=repo, llm_provider=mock_llm)

    response = service.ask(
        meeting_id="meet_chat_1",
        question="¿Qué base de datos se eligió?",
    )

    assert "SQLite FTS5" in response


def test_meeting_chat_service_non_existent_meeting(repo):
    mock_llm = MockLLMAdapter()
    service = MeetingChatService(repository=repo, llm_provider=mock_llm)

    with pytest.raises(ValueError, match="no encontrada"):
        service.ask(meeting_id="meet_inexistente", question="¿Hola?")
