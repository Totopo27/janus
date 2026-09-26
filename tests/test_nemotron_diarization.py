import os
import wave
import pytest
from janus.domain.models import AudioChunk
from janus.services.nemotron_diarization_service import NemotronDiarizationService, NemotronDiarizationReport


@pytest.fixture
def sample_audio_chunk():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(project_root, "assets", "models", "test_wavs", "0.wav")
    if not os.path.exists(wav_path):
        pytest.skip("Test audio 0.wav not found")

    with wave.open(wav_path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()

    return AudioChunk(data=data, sample_rate=sample_rate)


def test_nemotron_service_initialization():
    service = NemotronDiarizationService()
    assert service.sample_rate == 16000
    assert service.subsampling_factor == 8
    resolved = service._resolve_model_path()
    assert resolved is not None and os.path.exists(resolved)


def test_nemotron_service_analyze_monologue(sample_audio_chunk):
    service = NemotronDiarizationService()
    report = service.analyze(sample_audio_chunk)

    assert isinstance(report, NemotronDiarizationReport)
    assert report.num_speakers == 1
    assert report.is_monologue is True
    assert report.has_overlap is False
    assert len(report.segments) > 0
    assert report.segments[0].speaker_index == 1
    assert report.segments[0].confidence > 0.8
    assert "monologue" in report.acoustic_hint.lower()


def test_nemotron_service_empty_audio():
    service = NemotronDiarizationService()
    empty_chunk = AudioChunk(data=b"", sample_rate=16000)
    report = service.analyze(empty_chunk)

    assert report.is_monologue is True
    assert len(report.segments) == 0
