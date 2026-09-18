import os
import pytest
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    MeetingSummary,
    ActionItem,
)
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_janus.db")
    return SqliteMeetingRepository(db_path=db_file)


def test_save_and_retrieve_meeting(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")

    meeting = Meeting(
        meeting_id="meet_001",
        title="Revisión de Arquitectura Janus",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )

    repo.save_meeting(meeting)

    retrieved = repo.get_meeting("meet_001")
    assert retrieved is not None
    assert retrieved.meeting_id == "meet_001"
    assert retrieved.title == "Revisión de Arquitectura Janus"
    assert retrieved.speaker_a.name == "Carlos"
    assert retrieved.speaker_b.name == "Alice"
    assert retrieved.status == "active"
    assert len(retrieved.turns) == 0


def test_save_turns_and_retrieve_ordered(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")
    meeting = Meeting(meeting_id="meet_002", title="Sprint Planning", speaker_a=speaker_a, speaker_b=speaker_b)
    repo.save_meeting(meeting)

    # Add turns
    turn1 = ConversationTurn(
        turn_id="t_1",
        session_id="meet_002",
        speaker_id="spk_carlos",
        original_transcription=TranscriptionResult("Hola Alice, ¿cómo va el sprint?", "es", 0.0, 1.5),
        translation=TranslationResult("Hola Alice, ¿cómo va el sprint?", "es", "Hi Alice, how is the sprint going?", "en", 45.0),
    )
    turn2 = ConversationTurn(
        turn_id="t_2",
        session_id="meet_002",
        speaker_id="spk_alice",
        original_transcription=TranscriptionResult("Going great Carlos, ready to deploy", "en", 2.0, 3.8),
        translation=TranslationResult("Going great Carlos, ready to deploy", "en", "Va genial Carlos, listos para desplegar", "es", 38.0),
    )

    repo.save_turn("meet_002", turn1)
    repo.save_turn("meet_002", turn2)

    retrieved = repo.get_meeting("meet_002")
    assert len(retrieved.turns) == 2
    assert retrieved.turns[0].turn_id == "t_1"
    assert retrieved.turns[0].original_transcription.text == "Hola Alice, ¿cómo va el sprint?"
    assert retrieved.turns[1].turn_id == "t_2"
    assert retrieved.turns[1].translation.translated_text == "Va genial Carlos, listos para desplegar"


def test_save_summary_and_action_items(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")
    meeting = Meeting(meeting_id="meet_003", title="Acuerdo Comercial", speaker_a=speaker_a, speaker_b=speaker_b)
    repo.save_meeting(meeting)

    summary = MeetingSummary(
        executive_summary="Reunión bilingüe para acordar el despliegue del sistema.",
        key_points=[
            "Se aprobó la arquitectura hexagonal.",
            "Supertonic y Sherpa correrán en CPU local.",
        ],
        action_items=[
            ActionItem(assignee="Carlos", task="Preparar entorno de pruebas", due_hint="Viernes"),
            ActionItem(assignee="Alice", task="Revisar documentación de API", due_hint="Lunes"),
        ],
    )

    repo.save_summary("meet_003", summary)

    retrieved = repo.get_meeting("meet_003")
    assert retrieved.status == "completed"
    assert retrieved.summary is not None
    assert retrieved.summary.executive_summary == "Reunión bilingüe para acordar el despliegue del sistema."
    assert len(retrieved.summary.key_points) == 2
    assert len(retrieved.summary.action_items) == 2
    assert retrieved.summary.action_items[0].assignee == "Carlos"
    assert retrieved.summary.action_items[0].due_hint == "Viernes"


def test_list_and_delete_meetings(repo):
    speaker_a = SpeakerProfile(speaker_id="a", name="A", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="b", name="B", native_language="en")

    repo.save_meeting(Meeting(meeting_id="m1", title="Meeting 1", speaker_a=speaker_a, speaker_b=speaker_b))
    repo.save_meeting(Meeting(meeting_id="m2", title="Meeting 2", speaker_a=speaker_a, speaker_b=speaker_b))

    meetings = repo.list_meetings()
    assert len(meetings) == 2

    assert repo.delete_meeting("m1") is True
    assert repo.get_meeting("m1") is None
    assert len(repo.list_meetings()) == 1
