import pytest
import json
from unittest.mock import Mock
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    MeetingSummary,
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
    assert summary.executive_summary == ""
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


def test_no_commitments_does_not_create_summary_or_follow_up_task(service):
    notes_service, repo = service
    meeting = Meeting(
        meeting_id="greeting",
        title="Greeting",
        speaker_a=SpeakerProfile("a", "Alice", "en"),
        speaker_b=SpeakerProfile("b", "Bob", "es"),
    )
    repo.save_meeting(meeting)
    repo.save_turn("greeting", ConversationTurn(
        turn_id="hello",
        session_id="greeting",
        speaker_id="a",
        original_transcription=TranscriptionResult("Hello, how are you today?", "en"),
        translation=TranslationResult(
            "Hello, how are you today?", "en", "Hola, ¿cómo estás hoy?", "es",
        ),
    ))

    summary = notes_service.generate_notes_and_finalize("greeting")

    assert summary.executive_summary == ""
    assert summary.action_items == []
    assert summary.key_points == ["Alice: Hello, how are you today?"]


@pytest.mark.parametrize("payload", [
    "not json",
    "[]",
    '{"executive_summary": "A fabricated partial summary"}',
    '{"executive_summary": null, "key_points": [], "action_items": []}',
    '{"executive_summary": " ", "key_points": [], "action_items": []}',
    '{"executive_summary": 42, "key_points": [], "action_items": []}',
    '{"executive_summary": "Draft ready", "key_points": "Draft ready", "action_items": []}',
    '{"executive_summary": "Draft ready", "key_points": [42], "action_items": []}',
    '{"executive_summary": "Draft ready", "key_points": [" "], "action_items": []}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": null}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": ["Review"]}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": [{}]}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": [{"assignee": "Alice"}]}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": [{"assignee": "", "task": "Review"}]}',
    '{"executive_summary": "Draft ready", "key_points": [], "action_items": [{"assignee": "Alice", "task": "Review", "due_hint": 42}]}',
])
def test_invalid_llm_output_is_not_saved_as_generated_notes(recorded_service, payload):
    notes_service, _ = recorded_service
    notes_service.llm_provider = Mock()
    notes_service.llm_provider.generate.return_value = payload

    summary = notes_service.generate_notes_and_finalize("recorded")

    assert summary.summary_status == "unavailable"
    assert summary.degraded_reason == "invalid_llm_response"
    assert summary.executive_summary == ""
    assert summary.key_points == ["Alice: The draft is ready for review."]
    assert summary.action_items == []


@pytest.fixture
def recorded_service(service):
    notes_service, repo = service
    repo.save_meeting(Meeting(
        meeting_id="recorded",
        title="Recorded meeting",
        speaker_a=SpeakerProfile("a", "Alice", "en"),
        speaker_b=SpeakerProfile("b", "Bob", "es"),
    ))
    repo.save_turn("recorded", ConversationTurn(
        turn_id="recorded-turn",
        session_id="recorded",
        speaker_id="a",
        original_transcription=TranscriptionResult("The draft is ready for review.", "en"),
        translation=TranslationResult(
            "The draft is ready for review.", "en",
            "El borrador está listo para revisión.", "es",
        ),
    ))
    return notes_service, repo


@pytest.mark.parametrize("provider_available", [False, True])
def test_unavailable_synthesis_has_explicit_reason(recorded_service, provider_available):
    notes_service, _ = recorded_service
    if provider_available:
        notes_service.llm_provider = Mock()
        notes_service.llm_provider.generate.side_effect = RuntimeError("secret connection details")

    summary = notes_service.generate_notes_and_finalize("recorded")

    assert summary.summary_status == "unavailable"
    assert summary.degraded_reason == ("llm_error" if provider_available else "no_llm_provider")
    assert summary.executive_summary == ""
    assert summary.action_items == []


@pytest.mark.parametrize("fenced", [False, True])
def test_valid_llm_notes_allow_no_commitments(recorded_service, fenced):
    notes_service, _ = recorded_service
    payload = json.dumps({
        "executive_summary": "The draft is ready for review.",
        "key_points": [],
        "action_items": [],
    })
    notes_service.llm_provider = Mock()
    notes_service.llm_provider.generate.return_value = f"```json\n{payload}\n```" if fenced else payload

    summary = notes_service.generate_notes_and_finalize("recorded")

    assert summary.summary_status == "generated"
    assert summary.degraded_reason is None
    assert summary.executive_summary == "The draft is ready for review."
    assert summary.key_points == []
    assert summary.action_items == []


@pytest.mark.parametrize("blank_turn", [False, True])
def test_empty_transcript_is_not_sent_for_synthesis(service, blank_turn):
    notes_service, repo = service
    repo.save_meeting(Meeting(
        "empty", "Empty meeting",
        SpeakerProfile("a", "Alice", "en"), SpeakerProfile("b", "Bob", "es"),
    ))
    if blank_turn:
        repo.save_turn("empty", ConversationTurn(
            "blank", "empty", "a", TranscriptionResult(" \n", "en"),
            TranslationResult(" \n", "en", "", "es"),
        ))
    notes_service.llm_provider = Mock()
    notes_service.llm_provider.generate.side_effect = AssertionError("Empty transcript sent to LLM")

    summary = notes_service.generate_notes_and_finalize("empty")

    assert summary.summary_status == "empty"
    assert summary.degraded_reason is None
    assert summary.action_items == []
    assert summary.key_points == []
    assert repo.get_meeting("empty").status == "completed"


@pytest.mark.parametrize("export_method", ["export_as_markdown", "export_as_txt"])
@pytest.mark.parametrize("status, warning", [
    ("unavailable", "Síntesis automática no disponible"),
    ("legacy", "Estado de generación desconocido"),
    ("generated", None),
    ("empty", None),
])
def test_exports_distinguish_unavailable_and_legacy_notes(recorded_service, export_method, status, warning):
    notes_service, repo = recorded_service
    repo.save_summary("recorded", MeetingSummary(
        executive_summary="Manual summary" if status == "unavailable" else "",
        key_points=["Alice: The draft is ready for review."],
        summary_status=status,
        degraded_reason="no_llm_provider" if status == "unavailable" else None,
    ))

    exported = getattr(notes_service, export_method)("recorded")

    if warning:
        assert warning in exported
    else:
        assert "Síntesis automática no disponible" not in exported
        assert "Estado de generación desconocido" not in exported
    assert "Alice: The draft is ready for review." in exported
    if status == "unavailable":
        assert "Manual summary" in exported


@pytest.mark.parametrize("source_lang, text, translated_text", [
    ("en", "The draft is ready for review.", "Me encargo de enviar el borrador mañana"),
    ("es", "El borrador está listo para revisión.", "I will send the draft tomorrow"),
])
def test_fallback_does_not_extract_commitments_added_by_translation(
    recorded_service, source_lang, text, translated_text,
):
    notes_service, repo = recorded_service
    repo.save_turn("recorded", ConversationTurn(
        "recorded-turn", "recorded", "a", TranscriptionResult(text, source_lang),
        TranslationResult(text, source_lang, translated_text, "es" if source_lang == "en" else "en"),
    ))

    summary = notes_service.generate_notes_and_finalize("recorded")

    assert summary.action_items == []
