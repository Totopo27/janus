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
        self._is_fallback = False

    def _init_recognizer(self) -> None:
        if self._recognizer is not None or self._is_fallback:
            return

        try:
            import sherpa_onnx
            logger.info("Initializing Sherpa-ONNX offline recognizer...")

            # Auto-detect default model paths in assets/models/whisper relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            base_dir = os.path.join(project_root, "assets", "models", "whisper")

            enc = self.whisper_encoder or os.environ.get("WHISPER_ENCODER")
            if not enc or not os.path.exists(enc):
                for candidate in ["tiny-encoder.int8.onnx", "tiny-encoder.onnx"]:
                    p = os.path.join(base_dir, candidate)
                    if os.path.exists(p):
                        enc = p
                        break

            dec = self.whisper_decoder or os.environ.get("WHISPER_DECODER")
            if not dec or not os.path.exists(dec):
                for candidate in ["tiny-decoder.int8.onnx", "tiny-decoder.onnx"]:
                    p = os.path.join(base_dir, candidate)
                    if os.path.exists(p):
                        dec = p
                        break

            tok = self.tokens or os.environ.get("WHISPER_TOKENS")
            if not tok or not os.path.exists(tok):
                p = os.path.join(base_dir, "tiny-tokens.txt")
                if os.path.exists(p):
                    tok = p

            if enc and os.path.exists(enc) and dec and os.path.exists(dec) and tok and os.path.exists(tok):
                recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=enc,
                    decoder=dec,
                    tokens=tok,
                    language="",  # Auto-detect language natively from incoming speech
                    num_threads=self.num_threads,
                )
                self._recognizer = recognizer
                logger.info(f"Sherpa-ONNX Whisper recognizer initialized successfully from {os.path.basename(enc)}.")
            elif self.encoder and os.path.exists(self.encoder):
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
                logger.warning(f"No valid Sherpa-ONNX model files found at {base_dir}. STT operating without model.")
                self._is_fallback = True
        except Exception as e:
            logger.warning(f"Could not load Sherpa-ONNX native recognizer: {e}.")
            self._is_fallback = True

    def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        self._init_recognizer()

        if audio.is_empty:
            return TranscriptionResult(text="", language=language or "es")

        # If native recognizer is loaded
        if self._recognizer is not None and not self._is_fallback:
            try:
                import numpy as np
                samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
                stream = self._recognizer.create_stream()
                stream.accept_waveform(audio.sample_rate, samples)
                self._recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                detected_lang = getattr(stream.result, "lang", None) or language or "es"
                
                # Filter out background noise artifacts
                if text.lower() in ["[music]", "[musica]", "[applause]", "(music)", "(musica)", ""]:
                    return TranscriptionResult(
                        text="",
                        language=detected_lang,
                        start_time=0.0,
                        end_time=audio.duration_seconds,
                        confidence=0.0,
                    )

                logger.info(f"Sherpa-ONNX Whisper recognized '{text}' (detected language: {detected_lang})")
                return TranscriptionResult(
                    text=text,
                    language=detected_lang,
                    start_time=0.0,
                    end_time=audio.duration_seconds,
                    confidence=0.95,
                )
            except Exception as e:
                logger.error(f"Sherpa-ONNX stream decoding error: {e}")

        # Return empty result when no speech or model unavailable - never inject fake hallucinated dialogue
        return TranscriptionResult(
            text="",
            language=language or "es",
            start_time=0.0,
            end_time=audio.duration_seconds,
            confidence=0.0,
        )
