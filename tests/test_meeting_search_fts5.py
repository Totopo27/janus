import pytest
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    SearchResult,
)
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_janus_fts.db")
    return SqliteMeetingRepository(db_path=db_file)


def test_fts5_search_original_and_translated_text(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_carlos", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_alice", name="Alice", native_language="en")

    meeting = Meeting(
        meeting_id="meet_fts_1",
        title="Discusión de Arquitectura",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        topic_key="proyectos/janus/core",
    )
    repo.save_meeting(meeting)

    turn1 = ConversationTurn(
        turn_id="t_fts_1",
        session_id="meet_fts_1",
        speaker_id="spk_carlos",
        original_transcription=TranscriptionResult("Debemos implementar la arquitectura hexagonal", "es"),
        translation=TranslationResult(
            "Debemos implementar la arquitectura hexagonal",
            "es",
            "We must implement hexagonal architecture",
            "en",
        ),
    )
    turn2 = ConversationTurn(
        turn_id="t_fts_2",
        session_id="meet_fts_1",
        speaker_id="spk_alice",
        original_transcription=TranscriptionResult("I agree, decoupling the domain from adapters is key", "en"),
        translation=TranslationResult(
            "I agree, decoupling the domain from adapters is key",
            "en",
            "De acuerdo, desacoplar el dominio de los adaptadores es clave",
            "es",
        ),
    )
    repo.save_turn("meet_fts_1", turn1)
    repo.save_turn("meet_fts_1", turn2)

    # Search in Spanish original text
    results_es = repo.search_turns("arquitectura")
    assert len(results_es) == 1
    assert results_es[0].turn_id == "t_fts_1"
    assert "arquitectura" in results_es[0].snippet.lower()
    assert results_es[0].topic_key == "proyectos/janus/core"

    # Search in English original text
    results_en = repo.search_turns("decoupling")
    assert len(results_en) == 1
    assert results_en[0].turn_id == "t_fts_2"

    # Search in translated text
    results_trans = repo.search_turns("desacoplar")
    assert len(results_trans) == 1
    assert results_trans[0].turn_id == "t_fts_2"


def test_fts5_search_with_topic_key_filter(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_1", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_2", name="Alice", native_language="en")

    meeting1 = Meeting(
        meeting_id="m_janus",
        title="Janus S2ST",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        topic_key="proyectos/janus",
    )
    meeting2 = Meeting(
        meeting_id="m_salonero",
        title="Salonero Bot",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        topic_key="proyectos/salonero",
    )
    repo.save_meeting(meeting1)
    repo.save_meeting(meeting2)

    turn_janus = ConversationTurn(
        turn_id="t_j",
        session_id="m_janus",
        speaker_id="spk_1",
        original_transcription=TranscriptionResult("Revisemos el despliegue del pipeline", "es"),
        translation=TranslationResult("Revisemos el despliegue del pipeline", "es", "Let's review the pipeline deployment", "en"),
    )
    turn_salonero = ConversationTurn(
        turn_id="t_s",
        session_id="m_salonero",
        speaker_id="spk_1",
        original_transcription=TranscriptionResult("El despliegue del bot de pedidos", "es"),
        translation=TranslationResult("El despliegue del bot de pedidos", "es", "The ordering bot deployment", "en"),
    )
    repo.save_turn("m_janus", turn_janus)
    repo.save_turn("m_salonero", turn_salonero)

    # Search without filter: should match both
    all_results = repo.search_turns("despliegue")
    assert len(all_results) == 2

    # Search with specific topic_key filter
    janus_results = repo.search_turns("despliegue", topic_key="proyectos/janus")
    assert len(janus_results) == 1
    assert janus_results[0].meeting_id == "m_janus"

    # Search with hierarchical prefix topic_key
    prefix_results = repo.search_turns("despliegue", topic_key="proyectos")
    assert len(prefix_results) == 2


def test_fts5_search_handles_special_characters_gracefully(repo):
    speaker_a = SpeakerProfile(speaker_id="spk_1", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_2", name="Alice", native_language="en")
    m = Meeting(meeting_id="m_special", title="Special characters", speaker_a=speaker_a, speaker_b=speaker_b)
    repo.save_meeting(m)
    repo.save_turn("m_special", ConversationTurn(
        turn_id="t_spec",
        session_id="m_special",
        speaker_id="spk_1",
        original_transcription=TranscriptionResult("¿Qué tal? ¡Todo bien! 100% éxito", "es"),
        translation=TranslationResult("¿Qué tal? ¡Todo bien! 100% éxito", "es", "What's up? All good! 100% success", "en"),
    ))

    # Should not raise sqlite3.OperationalError on punctuation or unmatched quotes
    res1 = repo.search_turns('¿qué tal?')
    assert len(res1) >= 1

    res2 = repo.search_turns('"unmatched quote')
    assert isinstance(res2, list)

    res3 = repo.search_turns('')
    assert res3 == []


def test_fts_snippets_never_include_active_html(repo):
    speaker_a = SpeakerProfile("a", "Alice", "en")
    speaker_b = SpeakerProfile("b", "Bob", "es")
    repo.save_meeting(Meeting("xss", "XSS", speaker_a, speaker_b))
    malicious = '<img src=x onerror="globalThis.pwned=1"> agenda'
    repo.save_turn("xss", ConversationTurn(
        turn_id="xss-turn",
        session_id="xss",
        speaker_id="a",
        original_transcription=TranscriptionResult(malicious, "en"),
        translation=TranslationResult(malicious, "en", malicious, "es"),
    ))

    snippet = repo.search_turns("agenda")[0].snippet
    assert "<b>" not in snippet
    assert malicious in snippet


def test_replacing_turn_updates_single_fts_row(repo):
    speaker_a = SpeakerProfile("a", "Alice", "en")
    speaker_b = SpeakerProfile("b", "Bob", "es")
    repo.save_meeting(Meeting("replace", "Replace", speaker_a, speaker_b))

    def make_turn(text):
        return ConversationTurn(
            turn_id="same-turn",
            session_id="replace",
            speaker_id="a",
            original_transcription=TranscriptionResult(text, "en"),
            translation=TranslationResult(text, "en", text, "es"),
        )

    repo.save_turn("replace", make_turn("old phrase"))
    repo.save_turn("replace", make_turn("new phrase"))
    assert repo.search_turns("old") == []
    assert len(repo.search_turns("new")) == 1
