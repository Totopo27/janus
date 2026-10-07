import json
import pytest
from unittest.mock import MagicMock
from janus.domain.models import Meeting, ConversationTurn, TranscriptionResult, TranslationResult, SpeakerProfile
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.llm.factory import LLMProviderFactory
from janus.services.meeting_notes_service import MeetingNotesService


def test_gate3_llm_factory_instantiation():
    """
    Validates Gate 3:
    LLMProviderFactory creates Ollama (local RTX 3060), Gemini, Claude and Mock providers seamlessly.
    """
    factory = LLMProviderFactory()

    # 1. Mock
    mock_prov = factory.create("mock", default_response="Prueba exitosa")
    assert mock_prov.provider_name == "mock"
    assert mock_prov.generate("test") == "Prueba exitosa"

    # 2. Ollama
    ollama_prov = factory.create("ollama", base_url="http://localhost:11434", model="qwen2.5:3b")
    assert ollama_prov.provider_name == "ollama"

    # 3. Gemini
    gemini_prov = factory.create("gemini", api_key="dummy_gemini_key")
    assert gemini_prov.provider_name == "gemini"

    # 4. Claude
    claude_prov = factory.create("claude", api_key="dummy_claude_key")
    assert claude_prov.provider_name == "claude"


def test_gate3_structured_interview_minutes_synthesis(tmp_path):
    """
    Validates Gate 3:
    Generating structured meeting minutes using an LLM adapter
    produces a clean executive summary, key points, and action items with owners and due dates.
    """
    db_path = str(tmp_path / "gate3_test.db")
    repo = SqliteMeetingRepository(db_path=db_path)

    # Mock LLM returning structured JSON
    mock_llm = MagicMock()
    mock_llm.provider_name = "ollama"
    mock_llm.generate.return_value = json.dumps({
        "executive_summary": "Entrevista técnica sobre la nueva arquitectura de Janus.",
        "key_points": [
            "Se definió la arquitectura con compuertas de validación.",
            "Se implementó el filtro anti-fantasmas en diarización acústica.",
            "Se probó el soporte multi-hablante y solapamiento."
        ],
        "action_items": [
            {"assignee": "Carlos", "task": "Preparar la interfaz web móvil", "due_hint": "Mañana"},
            {"assignee": "Dra. Elena", "task": "Revisar los benchmarks de latencia", "due_hint": "Viernes"}
        ]
    })

    notes_service = MeetingNotesService(repository=repo, llm_provider=mock_llm)

    spk_a = SpeakerProfile(speaker_id="carlos", name="Carlos", native_language="es")
    spk_b = SpeakerProfile(speaker_id="elena", name="Dra. Elena", native_language="es")
    meeting = Meeting(
        meeting_id="interview_gate3",
        title="Entrevista Arquitectura Janus",
        speaker_a=spk_a,
        speaker_b=spk_b,
    )
    repo.save_meeting(meeting)

    turn1 = ConversationTurn(
        turn_id="t1",
        session_id="interview_gate3",
        speaker_id="carlos",
        original_transcription=TranscriptionResult(text="¿Cómo estructuramos la fase 3?", language="es"),
        translation=TranslationResult(source_text="¿Cómo estructuramos la fase 3?", source_lang="es", translated_text="How do we structure phase 3?", target_lang="en"),
    )
    repo.save_turn("interview_gate3", turn1)

    summary = notes_service.generate_notes_and_finalize("interview_gate3")
    assert summary is not None
    assert "Entrevista técnica" in summary.executive_summary
    assert len(summary.key_points) == 3
    assert len(summary.action_items) == 2
    assert summary.action_items[0].assignee == "Carlos"
    assert summary.action_items[0].due_hint == "Mañana"

    # Verify Markdown export contains the structured minutes
    md = notes_service.export_as_markdown("interview_gate3")
    assert "Entrevista Arquitectura Janus" in md
    assert "Carlos" in md
    assert "Dra. Elena" in md
