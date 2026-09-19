import io
import logging
import os
from typing import Optional
from janus.domain.models import SynthesisResult
from janus.ports.tts_port import ISpeechSynthesizer

logger = logging.getLogger(__name__)


class SupertonicTtsAdapter(ISpeechSynthesizer):
    """
    On-device, ultra-fast Text-to-Speech adapter powered by Supertonic ONNX Runtime.
    Synthesizes studio-grade 44.1 kHz WAV on CPU in ~120 ms.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        sample_rate: int = 44100,
        auto_download: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.auto_download = auto_download
        self._engine = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return

        try:
            import supertonic
            if not self.model_dir or not os.path.isdir(self.model_dir):
                raise RuntimeError("A valid local Supertonic model directory is required")
            logger.info("Initializing Supertonic TTS engine...")
            self._engine = supertonic.TTS(
                model_dir=self.model_dir,
                auto_download=self.auto_download,
            )
            logger.info("Supertonic engine loaded successfully.")
        except Exception as e:
            logger.error("Supertonic initialization failed (%s)", type(e).__name__)
            raise RuntimeError("Supertonic is not configured or could not be loaded") from None

    def synthesize(
        self,
        text: str,
        language: str = "es",
        voice_style: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesizes text into 44.1 kHz 16-bit WAV bytes.
        """
        self._ensure_loaded()
        clean_text = text.strip()
        if not clean_text:
            return SynthesisResult(
                audio_bytes=b"",
                sample_rate=self.sample_rate,
                duration_seconds=0.0,
                format="wav",
                voice_id=voice_style or "default",
            )

        try:
            import soundfile as sf
            voice_name = voice_style if voice_style and voice_style != "default" else "M1"
            style = self._engine.get_voice_style(voice_name)
            audio_wave, duration_value = self._engine.synthesize(
                clean_text,
                voice_style=style,
                lang=language,
            )
            duration = float(duration_value[0])

            buffer = io.BytesIO()
            sf.write(buffer, audio_wave.squeeze(), self.sample_rate, format="WAV", subtype="PCM_16")
            audio_bytes = buffer.getvalue()

            return SynthesisResult(
                audio_bytes=audio_bytes,
                sample_rate=self.sample_rate,
                duration_seconds=duration,
                format="wav",
                voice_id=voice_style or "default",
            )
        except Exception as e:
            logger.error("Supertonic synthesis failed (%s)", type(e).__name__)
            raise RuntimeError("Speech synthesis failed") from None
