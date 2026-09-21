import os
import numpy as np
import pytest
import soundfile as sf

from janus.domain.models import AudioChunk, Session, SpeakerProfile
from janus.services.vad_segmenter import SileroVadSegmenter
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from unittest.mock import MagicMock


def test_vad_segmenter_init():
    segmenter = SileroVadSegmenter(threshold=0.6, min_silence_duration=0.6)
    assert segmenter.threshold == 0.6
    assert segmenter.min_silence_duration == 0.6
    assert segmenter.min_speech_duration == 0.3


def test_vad_segmenter_empty_chunk():
    segmenter = SileroVadSegmenter()
    empty = AudioChunk(data=b"", sample_rate=16000)
    result = segmenter.segment_audio(empty)
    assert len(result) == 1
    assert result[0].is_empty


def test_vad_segmenter_short_audio():
    segmenter = SileroVadSegmenter()
    # 0.5s of audio is under 1.0s threshold
    short_data = np.zeros(8000, dtype=np.int16).tobytes()
    chunk = AudioChunk(data=short_data, sample_rate=16000)
    result = segmenter.segment_audio(chunk)
    assert len(result) == 1
    assert result[0] == chunk


def test_vad_segmenter_real_speech_splitting():
    segmenter = SileroVadSegmenter()
    segmenter._init_vad()
    if segmenter._is_fallback:
        pytest.skip("Silero VAD model not found in assets; skipping real segmentation test")

    test_wav = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "models", "test_wavs", "0.wav"
    )
    if not os.path.exists(test_wav):
        pytest.skip("Test WAV file not found")

    audio_data, sr = sf.read(test_wav)
    pcm = (audio_data * 32767).astype(np.int16).tobytes()
    silence = np.zeros(int(sr * 1.2), dtype=np.int16).tobytes()

    # Concatenate: Speech 1 + 1.2s Silence + Speech 2
    combined_pcm = pcm + silence + pcm
    chunk = AudioChunk(data=combined_pcm, sample_rate=sr)

    segments = segmenter.segment_audio(chunk)
    assert len(segments) >= 2, f"Expected at least 2 segments, got {len(segments)}"
    for s in segments:
        assert s.duration_seconds > 0.5


def test_pipeline_orchestrator_slices_with_segmenter():
    import asyncio

    async def run_test():
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = MagicMock(text="Hola mundo", language="es", confidence=0.95)

        mock_mt = MagicMock()
        mock_mt.translate.return_value = MagicMock(
            source_text="Hola mundo", translated_text="Hello world", latency_ms=120.0
        )

        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = MagicMock(
            audio_bytes=b"dummy_wav", duration_seconds=1.2, sample_rate=16000, format="wav"
        )

        mock_segmenter = MagicMock()
        c1 = AudioChunk(data=b"\x00" * 32000, sample_rate=16000)
        c2 = AudioChunk(data=b"\x00" * 32000, sample_rate=16000)
        mock_segmenter.segment_audio.return_value = [c1, c2]

        orchestrator = PipelineOrchestrator(
            stt_engine=mock_stt,
            translation_engine=mock_mt,
            tts_engine=mock_tts,
            segmenter=mock_segmenter,
        )

        session = Session(
            session_id="test_sess",
            speaker_a=SpeakerProfile(speaker_id="speaker_1", name="Hablante 1", native_language="es"),
            speaker_b=SpeakerProfile(speaker_id="speaker_2", name="Hablante 2", native_language="es"),
        )

        # 4-second input chunk
        input_chunk = AudioChunk(data=b"\x00" * 64000, sample_rate=16000)
        turn = await orchestrator.process_turn(session, "local", input_chunk)

        mock_segmenter.segment_audio.assert_called_once_with(input_chunk)
        assert mock_stt.transcribe.call_count == 2
        assert turn is not None

    asyncio.run(run_test())
