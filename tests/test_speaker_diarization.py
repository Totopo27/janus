import numpy as np
import pytest
from janus.domain.models import AudioChunk
from janus.services.speaker_diarization_service import SpeakerDiarizationService


def test_diarization_init():
    service = SpeakerDiarizationService()
    assert service.num_threads == 2
    assert service.similarity_threshold == 0.65
    assert service.min_duration_seconds == 0.4


def test_diarization_empty_chunk():
    service = SpeakerDiarizationService()
    empty = AudioChunk(data=b"", sample_rate=16000)
    spk_id, spk_name, conf = service.identify_speaker(empty, "test_session", "fallback_id", "Fallback Name")
    assert spk_id == "fallback_id"
    assert spk_name == "Fallback Name"


def test_diarization_short_audio():
    service = SpeakerDiarizationService()
    # 0.1 second of audio (too short)
    short_data = np.zeros(1600, dtype=np.int16).tobytes()
    chunk = AudioChunk(data=short_data, sample_rate=16000)
    spk_id, spk_name, conf = service.identify_speaker(chunk, "test_session", "local", "Hablante Local")
    assert spk_id == "local"
    assert spk_name == "Hablante Local"


def test_diarization_session_reset():
    service = SpeakerDiarizationService()
    service._session_managers["sess_1"] = "dummy"
    service._session_speaker_names["sess_1"] = {"speaker_1": "Hablante 1"}
    service.reset_session("sess_1")
    assert "sess_1" not in service._session_managers
    assert "sess_1" not in service._session_speaker_names


def test_diarization_with_real_model_if_present():
    service = SpeakerDiarizationService()
    service._init_extractor()
    if service._is_fallback:
        pytest.skip("CAM++ ONNX model not found in environment; running in fallback mode")

    # Generate 1.0s of synthetic 120Hz voice audio
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    voice_1 = (0.5 * np.sin(2 * np.pi * 120 * t) * 32767).astype(np.int16).tobytes()
    chunk_1 = AudioChunk(data=voice_1, sample_rate=16000)

    spk_id_1, spk_name_1, conf_1 = service.identify_speaker(chunk_1, "session_live")
    assert spk_id_1 == "speaker_1"
    assert spk_name_1 == "Hablante 1"

    # Same voice should match speaker_1
    spk_id_1_again, _, conf_match = service.identify_speaker(chunk_1, "session_live")
    assert spk_id_1_again == "speaker_1"
    assert conf_match >= 0.65
