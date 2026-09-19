import asyncio
import pytest
from janus.domain.models import (
    AudioChunk,
    Session,
    SpeakerProfile,
)
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.services.pipeline_orchestrator import PipelineOrchestrator


def test_pipeline_orchestrator_full_flow():
    async def run_test():
        # Setup session
        speaker_a = SpeakerProfile(speaker_id="carlos", name="Carlos", native_language="es")
        speaker_b = SpeakerProfile(speaker_id="alice", name="Alice", native_language="en")
        session = Session(session_id="sess_001", speaker_a=speaker_a, speaker_b=speaker_b)

        # Setup adapters
        stt = MockSpeechRecognizer(predefined_text="Hola mundo", language="es")
        mt = MockTranslator({"Hola mundo": "Hello world"})
        tts = MockSpeechSynthesizer()
        broadcaster = WebSocketBroadcaster()

        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            broadcaster=broadcaster,
        )

        audio_chunk = AudioChunk(data=b"\x00\x01" * 8000, sample_rate=16000)  # 0.5s audio

        # Execute turn: Carlos speaks in Spanish
        turn = await orchestrator.process_turn(
            session=session,
            speaker_id="carlos",
            audio=audio_chunk,
        )

        assert turn is not None
        assert turn.speaker_id == "carlos"
        assert turn.original_transcription.text == "Hola mundo"
        assert turn.original_transcription.language == "es"
        assert turn.translation.translated_text == "Hello world"
        assert turn.translation.target_lang == "en"
        assert turn.synthesis is not None
        assert turn.synthesis.format == "wav"

        # Verify turn was recorded in session
        assert len(session.turns) == 1
        assert session.turns[0].turn_id == turn.turn_id

        # Verify adapter calls
        assert stt.transcribe_called_count == 1
        assert mt.translate_called_count == 1
        assert tts.synthesize_called_count == 1

    asyncio.run(run_test())


def test_pipeline_orchestrator_skips_empty_audio():
    async def run_test():
        speaker_a = SpeakerProfile(speaker_id="carlos", name="Carlos", native_language="es")
        speaker_b = SpeakerProfile(speaker_id="alice", name="Alice", native_language="en")
        session = Session(session_id="sess_002", speaker_a=speaker_a, speaker_b=speaker_b)

        stt = MockSpeechRecognizer()
        mt = MockTranslator()
        tts = MockSpeechSynthesizer()
        broadcaster = WebSocketBroadcaster()

        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            broadcaster=broadcaster,
        )

        empty_chunk = AudioChunk(data=b"")
        turn = await orchestrator.process_turn(
            session=session,
            speaker_id="carlos",
            audio=empty_chunk,
        )

        assert turn is None
        assert len(session.turns) == 0
        assert stt.transcribe_called_count == 0

    asyncio.run(run_test())


def test_storage_failure_does_not_publish_turn_to_session():
    class FailingStorage:
        def save_turn(self, meeting_id, turn):
            raise OSError("disk unavailable")

    async def run_test():
        speaker_a = SpeakerProfile("carlos", "Carlos", "es")
        speaker_b = SpeakerProfile("alice", "Alice", "en")
        session = Session("storage-failure", speaker_a, speaker_b)
        orchestrator = PipelineOrchestrator(
            stt_engine=MockSpeechRecognizer(),
            translation_engine=MockTranslator(),
            tts_engine=MockSpeechSynthesizer(),
            storage_repo=FailingStorage(),
        )
        with pytest.raises(OSError, match="disk unavailable"):
            await orchestrator.process_turn(
                session=session,
                speaker_id="carlos",
                audio=AudioChunk(data=b"\x00\x00" * 100),
            )
        assert session.turns == []

    asyncio.run(run_test())
