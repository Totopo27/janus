import os
import wave
import pytest
from janus.domain.models import AudioChunk
from janus.services.speaker_diarization_service import (
    SpeakerDiarizationService,
    DiarizationReport,
    DiarizationSegment,
)


def _load_sample_audio() -> AudioChunk:
    wav_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "assets", "models", "test_wavs", "0.wav"
    )
    if not os.path.exists(wav_path):
        pytest.skip(f"Test wav file not found at {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return AudioChunk(data=raw, sample_rate=sr)


def test_diarization_service_init_and_analyze_real_audio():
    service = SpeakerDiarizationService()
    audio = _load_sample_audio()

    report = service.analyze_segments(audio)
    assert isinstance(report, DiarizationReport)
    assert report.num_speakers >= 1
    # 0.wav has a single speaker
    assert report.is_monologue is True
    assert "monologue" in report.acoustic_hint.lower()
    assert len(report.segments) >= 1
    assert report.segments[0].start >= 0.0
    assert report.segments[0].end > report.segments[0].start


def test_diarization_service_empty_audio():
    service = SpeakerDiarizationService()
    empty_audio = AudioChunk(data=b"", sample_rate=16000)
    report = service.analyze_segments(empty_audio)
    assert isinstance(report, DiarizationReport)
    assert report.num_speakers == 1
    assert report.is_monologue is True


def test_diarization_service_fallback_mode():
    # Point to nonexistent models to force fallback
    service = SpeakerDiarizationService(
        model_path="nonexistent_campplus.onnx",
        segmentation_model_path="nonexistent_segmentation.onnx",
    )
    audio = AudioChunk(data=b"\x00\x01" * 8000, sample_rate=16000)
    report = service.analyze_segments(audio)
    assert isinstance(report, DiarizationReport)
    assert report.num_speakers == 1
    assert report.is_monologue is True


def test_pipeline_orchestrator_passes_acoustic_hint_to_fusion():
    import asyncio
    from unittest.mock import MagicMock
    from janus.domain.models import SpeakerProfile, Session
    from janus.services.pipeline_orchestrator import PipelineOrchestrator
    from tests.test_pipeline_orchestrator import MockSpeechRecognizer, MockTranslator, MockSpeechSynthesizer
    from janus.services.conversational_fusion_service import FusedTurn

    async def run_test():
        speaker_a = SpeakerProfile(speaker_id="speaker_1", name="Hablante 1", native_language="es")
        speaker_b = SpeakerProfile(speaker_id="speaker_2", name="Hablante 2", native_language="es")
        session = Session(session_id="sess_hint", speaker_a=speaker_a, speaker_b=speaker_b)

        stt = MockSpeechRecognizer(predefined_text="Hola mundo", language="es")
        mt = MockTranslator()
        tts = MockSpeechSynthesizer()

        mock_diarizer = MagicMock()
        mock_diarizer.analyze_segments.return_value = DiarizationReport(
            num_speakers=1,
            is_monologue=True,
            acoustic_hint="Acoustic diarizer confirmed a single speaker (monologue).",
        )
        mock_diarizer.identify_speaker.return_value = ("speaker_1", "Hablante 1", 0.95)

        mock_fusion = MagicMock()
        mock_fusion.fuse_and_translate.return_value = [
            FusedTurn(speaker_id="speaker_1", speaker_name="Hablante 1", original_text="Hola mundo", translated_text="Hello world")
        ]

        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            diarizer=mock_diarizer,
            fusion_service=mock_fusion,
        )

        audio = AudioChunk(data=b"\x00\x01" * 16000, sample_rate=16000)
        await orchestrator.process_turn(session, "speaker_1", audio)

        mock_fusion.fuse_and_translate.assert_called_once()
        _, kwargs = mock_fusion.fuse_and_translate.call_args
        assert kwargs["acoustic_hint"] == "Acoustic diarizer confirmed a single speaker (monologue)."

    asyncio.run(run_test())

