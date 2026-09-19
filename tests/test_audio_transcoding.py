import io

import av
import numpy as np
import pytest

from janus.adapters.stt.audio_transcoding import (
    AudioTranscodingError,
    transcode_to_pcm16,
)


def make_webm_opus(duration_seconds: int = 1) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48_000)
        stream.layout = "mono"
        samples = np.zeros((1, 48_000 * duration_seconds), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.sample_rate = 48_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def test_webm_opus_is_transcoded_to_mono_16khz_pcm16():
    pcm = transcode_to_pcm16(make_webm_opus(), "audio/webm;codecs=opus")
    assert len(pcm) == 16_000 * 2


def test_transcoder_rejects_unknown_formats_and_misaligned_pcm():
    with pytest.raises(AudioTranscodingError, match="Unsupported"):
        transcode_to_pcm16(b"not audio", "audio/mpeg")
    with pytest.raises(AudioTranscodingError, match="complete samples"):
        transcode_to_pcm16(b"\x00", "audio/pcm;rate=16000")
