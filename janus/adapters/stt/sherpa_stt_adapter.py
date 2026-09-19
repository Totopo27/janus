import logging
import os
from typing import Optional
from janus.domain.models import AudioChunk, TranscriptionResult
from janus.ports.stt_port import ISpeechRecognizer

logger = logging.getLogger(__name__)


class SherpaSttAdapter(ISpeechRecognizer):
    """
    On-device, offline Automatic Speech Recognition (STT) adapter
    powered by Sherpa-ONNX (Next-gen Kaldi / Whisper INT8 ONNX).
    """

    def __init__(
        self,
        tokens: Optional[str] = None,
        encoder: Optional[str] = None,
        decoder: Optional[str] = None,
        joiner: Optional[str] = None,
        whisper_encoder: Optional[str] = None,
        whisper_decoder: Optional[str] = None,
        model_type: str = "whisper",
        num_threads: int = 4,
    ) -> None:
        self.tokens = tokens
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.whisper_encoder = whisper_encoder
        self.whisper_decoder = whisper_decoder
        self.model_type = model_type
        self.num_threads = num_threads
        self._recognizer = None

    def _init_recognizer(self) -> None:
        if self._recognizer is not None:
            return

        try:
            import sherpa_onnx
            logger.info("Initializing Sherpa-ONNX offline recognizer...")

            # If Whisper ONNX model files are provided and exist:
            if self.whisper_encoder and os.path.exists(self.whisper_encoder):
                recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=self.whisper_encoder,
                    decoder=self.whisper_decoder,
                    tokens=self.tokens,
                    num_threads=self.num_threads,
                )
                self._recognizer = recognizer
                logger.info("Sherpa-ONNX Whisper recognizer initialized successfully.")
            elif self.encoder and os.path.exists(self.encoder):
                # Zipformer model
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    tokens=self.tokens,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    joiner=self.joiner,
                    num_threads=self.num_threads,
                )
                self._recognizer = recognizer
                logger.info("Sherpa-ONNX Transducer recognizer initialized successfully.")
            else:
                raise RuntimeError("Valid Sherpa-ONNX model files are required")
        except Exception as e:
            logger.error("Sherpa-ONNX recognizer initialization failed (%s)", type(e).__name__)
            raise RuntimeError("Sherpa-ONNX recognizer is not configured or could not be loaded") from None

    def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        self._init_recognizer()

        if audio.is_empty:
            return TranscriptionResult(text="", language=language or "es")
        if audio.sample_rate != 16000 or audio.channels != 1 or len(audio.data) % 2:
            raise ValueError("Sherpa requires mono 16 kHz PCM16 audio")

        try:
            import numpy as np
            samples = np.frombuffer(audio.data, dtype="<i2").astype(np.float32) / 32768.0
            stream = self._recognizer.create_stream()
            stream.accept_waveform(audio.sample_rate, samples)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            return TranscriptionResult(
                text=text,
                language=language or "es",
                start_time=0.0,
                end_time=audio.duration_seconds,
                confidence=0.95,
            )
        except Exception as e:
            logger.error("Sherpa-ONNX decoding failed (%s)", type(e).__name__)
            raise RuntimeError("Speech recognition failed") from None
