import pytest
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
)
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.services.meeting_notes_service import MeetingNotesService


@pytest.fixture
def service(tmp_path):
    repo = SqliteMeetingRepository(db_path=str(tmp_path / "notes_test.db"))
    return MeetingNotesService(repository=repo), repo


def test_generate_meeting_notes_and_export_markdown(service):
    notes_service, repo = service

    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")

    meeting = Meeting(
        meeting_id="meet_zoom_1",
        title="Discusión Técnica Janus S2ST",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    repo.save_meeting(meeting)

    turn1 = ConversationTurn(
        turn_id="t1",
        session_id="meet_zoom_1",
        speaker_id="spk_carlos",
        original_transcription=TranscriptionResult("Hola Alice, yo me comprometo a compilar los modelos ONNX", "es"),
        translation=TranslationResult("Hola Alice, yo me comprometo a compilar los modelos ONNX", "es", "Hi Alice, I commit to compiling the ONNX models", "en"),
    )
    turn2 = ConversationTurn(
        turn_id="t2",
        session_id="meet_zoom_1",
        speaker_id="spk_alice",
        original_transcription=TranscriptionResult("Awesome, I will review the API endpoints tomorrow", "en"),
        translation=TranslationResult("Awesome, I will review the API endpoints tomorrow", "en", "Genial, yo revisaré los endpoints de la API mañana", "es"),
    )

    repo.save_turn("meet_zoom_1", turn1)
    repo.save_turn("meet_zoom_1", turn2)

    # Generate notes & finalize
    summary = notes_service.generate_notes_and_finalize("meet_zoom_1")
    assert summary is not None
    assert len(summary.executive_summary) > 0
    assert len(summary.key_points) > 0
    assert len(summary.action_items) > 0

    # Verify meeting updated in repo
    updated = repo.get_meeting("meet_zoom_1")
    assert updated.status == "completed"
    assert updated.summary is not None

    # Export to markdown
    markdown = notes_service.export_as_markdown("meet_zoom_1")
    assert "# Minuta de Reunión: Discusión Técnica Janus S2ST" in markdown
    assert "Carlos (es)" in markdown
    assert "Alice (en)" in markdown
    assert "## Resumen Ejecutivo" in markdown
    assert "## Puntos Clave" in markdown
    assert "## Compromisos y Tareas (Action Items)" in markdown
    assert "## Transcripción Bilingüe Cronológica" in markdown
