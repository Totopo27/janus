import asyncio
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


def test_pipeline_orchestrator_multi_turn_fusion():
    from unittest.mock import MagicMock
    from janus.services.conversational_fusion_service import FusedTurn

    async def run_test():
        speaker_a = SpeakerProfile(speaker_id="speaker_1", name="Hablante 1", native_language="es")
        speaker_b = SpeakerProfile(speaker_id="speaker_2", name="Hablante 2", native_language="es")
        session = Session(session_id="sess_fused", speaker_a=speaker_a, speaker_b=speaker_b)

        stt = MockSpeechRecognizer(predefined_text="¿Vives aquí? Sí, vivo aquí.", language="es")
        mt = MockTranslator()
        tts = MockSpeechSynthesizer()
        broadcaster = WebSocketBroadcaster()

        mock_fusion = MagicMock()
        mock_fusion.fuse_and_translate.return_value = [
            FusedTurn(speaker_id="speaker_1", speaker_name="Hablante 1", original_text="¿Vives aquí?", translated_text="Do you live here?"),
            FusedTurn(speaker_id="speaker_2", speaker_name="Hablante 2", original_text="Sí, vivo aquí.", translated_text="Yes, I live here."),
        ]

        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            broadcaster=broadcaster,
            fusion_service=mock_fusion,
        )

        audio_chunk = AudioChunk(data=b"\x00\x01" * 16000, sample_rate=16000)
        last_turn = await orchestrator.process_turn(session, "speaker_1", audio_chunk)

        assert last_turn is not None
        assert last_turn.speaker_id == "speaker_2"
        assert len(session.turns) == 2
        assert session.turns[0].speaker_id == "speaker_1"
        assert session.turns[0].original_transcription.text == "¿Vives aquí?"
        assert session.turns[0].translation.translated_text == "Do you live here?"
        assert session.turns[1].speaker_id == "speaker_2"
        assert session.turns[1].original_transcription.text == "Sí, vivo aquí."
        assert session.turns[1].translation.translated_text == "Yes, I live here."

    asyncio.run(run_test())
