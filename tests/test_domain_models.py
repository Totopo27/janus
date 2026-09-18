import pytest
from datetime import datetime, timezone
from janus.domain.models import (
    AudioChunk,
    TranscriptionResult,
    TranslationResult,
    SynthesisResult,
    ConversationTurn,
    Session,
    SpeakerProfile,
)
from janus.domain.events import (
    DomainEvent,
    SpeechDetectedEvent,
    TranscriptionCompletedEvent,
    TranslationCompletedEvent,
    SynthesisCompletedEvent,
    TurnCompletedEvent,
)


def test_audio_chunk_creation_and_properties():
    data = b"\x00\x01\x00\x02" * 100
    chunk = AudioChunk(
        data=data,
        sample_rate=16000,
        channels=1,
        timestamp=1726668000.0,
    )
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.duration_seconds == len(data) / (16000 * 2)  # 16-bit PCM = 2 bytes per sample
    assert not chunk.is_empty


def test_empty_audio_chunk():
    empty_chunk = AudioChunk(data=b"", sample_rate=16000)
    assert empty_chunk.is_empty
    assert empty_chunk.duration_seconds == 0.0


def test_transcription_result_model():
    result = TranscriptionResult(
        text="Hola, ¿cómo estás?",
        language="es",
        start_time=0.0,
        end_time=1.5,
        confidence=0.98,
    )
    assert result.text == "Hola, ¿cómo estás?"
    assert result.language == "es"
    assert result.confidence == 0.98


def test_translation_result_model():
    res = TranslationResult(
        source_text="Hola, ¿cómo estás?",
        source_lang="es",
        translated_text="Hello, how are you?",
        target_lang="en",
        latency_ms=45.2,
    )
    assert res.translated_text == "Hello, how are you?"
    assert res.target_lang == "en"
    assert res.latency_ms > 0


def test_synthesis_result_model():
    audio_data = b"RIFF....WAVE"
    res = SynthesisResult(
        audio_bytes=audio_data,
        sample_rate=44100,
        duration_seconds=1.2,
        format="wav",
        voice_id="default",
    )
    assert res.sample_rate == 44100
    assert res.format == "wav"


def test_session_and_conversation_turn():
    speaker_a = SpeakerProfile(speaker_id="spk_a", name="Carlos", native_language="es")
    speaker_b = SpeakerProfile(speaker_id="spk_b", name="Alice", native_language="en")

    session = Session(
        session_id="sess_123",
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )

    assert session.session_id == "sess_123"
    assert len(session.turns) == 0

    turn = ConversationTurn(
        turn_id="turn_1",
        session_id="sess_123",
        speaker_id="spk_a",
        original_transcription=TranscriptionResult(
            text="Buenas tardes",
            language="es",
            start_time=0.0,
            end_time=1.0,
        ),
        translation=TranslationResult(
            source_text="Buenas tardes",
            source_lang="es",
            translated_text="Good afternoon",
            target_lang="en",
            latency_ms=30.0,
        ),
    )

    session.add_turn(turn)
    assert len(session.turns) == 1
    assert session.turns[0].speaker_id == "spk_a"
    assert session.turns[0].translation.translated_text == "Good afternoon"


def test_domain_events():
    event = TurnCompletedEvent(
        session_id="sess_123",
        turn_id="turn_1",
        speaker_id="spk_a",
        original_text="Hola",
        translated_text="Hello",
        source_lang="es",
        target_lang="en",
        created_at=datetime.now(timezone.utc),
    )
    assert event.session_id == "sess_123"
    assert event.original_text == "Hola"
    assert event.translated_text == "Hello"
