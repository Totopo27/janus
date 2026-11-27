import pytest
from janus.adapters.stt.audio_transcoding import (
    AudioTranscodingError,
    MAX_PCM_BYTES,
    transcode_to_pcm16,
)


def test_transcode_empty_bytes():
    result = transcode_to_pcm16(b"", format_hint="audio/webm")
    assert result == b""


def test_transcode_invalid_audio_fallback():
    # Garbage bytes that PyAV cannot decode should safely fallback to original bytes without crashing
    garbage = b"NOT_A_VALID_AUDIO_CONTAINER_12345"
    result = transcode_to_pcm16(garbage, format_hint="audio/webm")
    assert result == garbage


def test_transcode_fallback_rejects_oversized_payload():
    oversized = b"X" * (MAX_PCM_BYTES + 2)
    with pytest.raises(AudioTranscodingError, match="maximum supported duration"):
        transcode_to_pcm16(oversized, format_hint="audio/webm")
