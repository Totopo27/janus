from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
    """Raised when audio payload decoding or format validation fails."""
    pass


def _normalized_content_type(content_type: str) -> str:
    return content_type.lower().replace(" ", "")


def transcode_to_pcm16(
    audio_bytes: bytes,
    content_type: str = "audio/webm",
    sample_rate: int = TARGET_SAMPLE_RATE,
    format_hint: Optional[str] = None,
) -> bytes:
    """
    Decodes a browser audio container (WebM/Ogg/Opus or PCM) into mono 16 kHz signed 16-bit PCM bytes.
    """
    if not audio_bytes:
        return b""
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioTranscodingError(f"Only {TARGET_SAMPLE_RATE} Hz output sample rate is supported")

    actual_content_type = format_hint or content_type
    normalized_type = _normalized_content_type(actual_content_type)

    # Handle raw PCM inputs
    if normalized_type in {"audio/pcm", "audio/pcm;rate=16000", "audio/l16", "raw"}:
        if len(audio_bytes) % 2 != 0:
            raise AudioTranscodingError("PCM16 payload must contain an even number of bytes")
        if len(audio_bytes) > MAX_PCM_BYTES:
            raise AudioTranscodingError("Decoded audio exceeds maximum supported duration")
        return audio_bytes

    # Check supported container formats
    if normalized_type not in SUPPORTED_COMPRESSED_TYPES and not normalized_type.startswith("audio/"):
        raise AudioTranscodingError(f"Unsupported audio content type: {content_type}")

    output = bytearray()
    try:
        import av

        with av.open(io.BytesIO(audio_bytes), mode="r") as container:
            if not container.streams.audio:
                raise AudioTranscodingError("Audio container contains no audio streams")

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=sample_rate,
            )

            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    output.extend(converted.to_ndarray().astype("<i2", copy=False).tobytes())
                    if len(output) > MAX_PCM_BYTES:
                        raise AudioTranscodingError("Decoded audio exceeds maximum supported duration")

            for converted in resampler.resample(None):
                output.extend(converted.to_ndarray().astype("<i2", copy=False).tobytes())

    except AudioTranscodingError:
        raise
    except ImportError:
        logger.warning("PyAV ('av') package not available. Falling back to raw byte handling.")
        return audio_bytes
    except Exception as exc:
        logger.debug(f"PyAV transcoding failed: {exc}. Retaining raw audio bytes.")
        return audio_bytes

    if not output:
        raise AudioTranscodingError("Audio container decoded to 0 samples")

    return bytes(output)
