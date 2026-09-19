from __future__ import annotations

import io

import av


TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION_SECONDS = 120
MAX_PCM_BYTES = TARGET_SAMPLE_RATE * 2 * MAX_AUDIO_DURATION_SECONDS
SUPPORTED_COMPRESSED_TYPES = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/ogg",
    "audio/ogg;codecs=opus",
}


class AudioTranscodingError(ValueError):
    pass


def _normalized_content_type(content_type: str) -> str:
    return content_type.lower().replace(" ", "")


def transcode_to_pcm16(
    audio_bytes: bytes,
    content_type: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> bytes:
    """Decode a supported browser audio container into mono, signed PCM16."""
    if not audio_bytes:
        raise AudioTranscodingError("Audio payload is empty")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioTranscodingError("Only 16 kHz output is supported")

    normalized_type = _normalized_content_type(content_type)
    if normalized_type in {"audio/pcm", "audio/pcm;rate=16000", "audio/l16"}:
        if len(audio_bytes) % 2:
            raise AudioTranscodingError("PCM16 payload must contain complete samples")
        if len(audio_bytes) > MAX_PCM_BYTES:
            raise AudioTranscodingError("Decoded audio exceeds the maximum duration")
        return audio_bytes
    if normalized_type not in SUPPORTED_COMPRESSED_TYPES:
        raise AudioTranscodingError("Unsupported audio content type")

    output = bytearray()
    try:
        with av.open(io.BytesIO(audio_bytes), mode="r") as container:
            if not container.streams.audio:
                raise AudioTranscodingError("Audio container has no audio stream")
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=sample_rate,
            )
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    output.extend(converted.to_ndarray().astype("<i2", copy=False).tobytes())
                    if len(output) > MAX_PCM_BYTES:
                        raise AudioTranscodingError("Decoded audio exceeds the maximum duration")
            for converted in resampler.resample(None):
                output.extend(converted.to_ndarray().astype("<i2", copy=False).tobytes())
    except AudioTranscodingError:
        raise
    except Exception as exc:
        raise AudioTranscodingError("Audio payload could not be decoded") from exc

    if not output:
        raise AudioTranscodingError("Audio payload decoded to zero samples")
    return bytes(output)
