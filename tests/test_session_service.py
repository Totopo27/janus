import pytest
from janus.domain.models import SpeakerProfile, ConversationTurn, TranscriptionResult, TranslationResult
from janus.services.session_service import SessionService


def test_session_service_crud():
    service = SessionService()

    speaker_a = SpeakerProfile(speaker_id="spk_a", name="Elena", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_b", name="John", native_language="en")

    # 1. Create
    session = service.create_session(
        session_id="meeting_1",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    assert session.session_id == "meeting_1"
    assert session.speaker_a.name == "Elena"

    # 2. Get
    retrieved = service.get_session("meeting_1")
    assert retrieved is not None
    assert retrieved.session_id == "meeting_1"

    # 3. List
    sessions = service.list_sessions()
    assert len(sessions) == 1

    # 4. Export transcript
    turn = ConversationTurn(
        turn_id="t1",
        session_id="meeting_1",
        speaker_id="spk_a",
        original_transcription=TranscriptionResult("Hola", "es"),
        translation=TranslationResult("Hola", "es", "Hello", "en"),
    )
    session.add_turn(turn)

    transcript = service.export_transcript("meeting_1")
    assert len(transcript) == 1
    assert transcript[0]["original_text"] == "Hola"
    assert transcript[0]["translated_text"] == "Hello"

    # 5. Close / Delete
    service.close_session("meeting_1")
    assert service.get_session("meeting_1") is None
