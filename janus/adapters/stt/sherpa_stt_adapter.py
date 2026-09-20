import logging
import os
from typing import Optional
from janus.domain.models import AudioChunk, TranscriptionResult
from janus.ports.stt_port import ISpeechRecognizer

logger = logging.getLogger(__name__)


class SherpaSttAdapter(ISpeechRecognizer):
    """
    On-device, offline Automatic Speech Recognition (STT) adapter
    powered by Sherpa-ONNX (Whisper Base/Small INT8 ONNX).
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
        default_language: str = "es",
    ) -> None:
        self.tokens = tokens
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.whisper_encoder = whisper_encoder
        self.whisper_decoder = whisper_decoder
        self.model_type = model_type
        self.num_threads = num_threads
        self.default_language = default_language
        self._recognizers: dict[str, any] = {}
        self._resolved_enc = None
        self._resolved_dec = None
        self._resolved_tok = None
        self._is_fallback = False

    def _resolve_model_paths(self) -> None:
        if self._resolved_enc and self._resolved_dec and self._resolved_tok:
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        base_dir = os.path.join(project_root, "assets", "models", "whisper")

        # Explicit overrides
        enc = self.whisper_encoder or os.environ.get("WHISPER_ENCODER")
        dec = self.whisper_decoder or os.environ.get("WHISPER_DECODER")
        tok = self.tokens or os.environ.get("WHISPER_TOKENS")

        if enc and dec and tok and os.path.exists(enc) and os.path.exists(dec) and os.path.exists(tok):
            self._resolved_enc = enc
            self._resolved_dec = dec
            self._resolved_tok = tok
            return

        # Prioritized candidate search: Small -> Base -> Tiny
        candidate_triplets = [
            ("small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt"),
            ("small-encoder.onnx", "small-decoder.onnx", "small-tokens.txt"),
            ("base-encoder.int8.onnx", "base-decoder.int8.onnx", "base-tokens.txt"),
            ("base-encoder.onnx", "base-decoder.onnx", "base-tokens.txt"),
            ("tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx", "tiny-tokens.txt"),
            ("tiny-encoder.onnx", "tiny-decoder.onnx", "tiny-tokens.txt"),
        ]

        for cand_enc, cand_dec, cand_tok in candidate_triplets:
            p_enc = os.path.join(base_dir, cand_enc)
            p_dec = os.path.join(base_dir, cand_dec)
            p_tok = os.path.join(base_dir, cand_tok)
            if os.path.exists(p_enc) and os.path.exists(p_dec) and os.path.exists(p_tok):
                self._resolved_enc = p_enc
                self._resolved_dec = p_dec
                self._resolved_tok = p_tok
                logger.info(f"Sherpa-ONNX auto-selected Whisper model: {cand_enc}")
                return

    def _get_recognizer(self, language: Optional[str] = None):
        if self._is_fallback:
            return None

        # Normalize language key
        lang_key = (language or self.default_language or "es").strip().lower()
        if lang_key in ["auto", "none", ""]:
            lang_key = ""
        elif lang_key.startswith("es"):
            lang_key = "es"
        elif lang_key.startswith("en"):
            lang_key = "en"

        if lang_key in self._recognizers:
            return self._recognizers[lang_key]

        try:
            import sherpa_onnx
            self._resolve_model_paths()

            if self._resolved_enc and self._resolved_dec and self._resolved_tok:
                recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=self._resolved_enc,
                    decoder=self._resolved_dec,
                    tokens=self._resolved_tok,
                    language=lang_key,
                    num_threads=self.num_threads,
                )
                self._recognizers[lang_key] = recognizer
                logger.info(
                    f"Sherpa-ONNX Whisper recognizer initialized from {os.path.basename(self._resolved_enc)} "
                    f"[language='{lang_key or 'auto'}']"
                )
                return recognizer
            elif self.encoder and os.path.exists(self.encoder):
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    tokens=self.tokens,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    joiner=self.joiner,
                    num_threads=self.num_threads,
                )
                self._recognizers["transducer"] = recognizer
                return recognizer
            else:
                logger.warning("No valid Sherpa-ONNX model files found. STT operating in fallback mode.")
                self._is_fallback = True
                return None
        except Exception as e:
            logger.warning(f"Could not load Sherpa-ONNX native recognizer: {e}.")
            self._is_fallback = True
            return None

    def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        effective_lang = language if language is not None else self.default_language
        recognizer = self._get_recognizer(effective_lang)

        if audio.is_empty:
            return TranscriptionResult(text="", language=effective_lang or "es")

        # If native recognizer is loaded
        if recognizer is not None and not self._is_fallback:
            try:
                import numpy as np
                samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
                stream = recognizer.create_stream()
                stream.accept_waveform(audio.sample_rate, samples)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                detected_lang = getattr(stream.result, "lang", None) or effective_lang or "es"

                # Filter out background noise artifacts
                if text.lower() in ["[music]", "[musica]", "[applause]", "(music)", "(musica)", ""]:
                    return TranscriptionResult(
                        text="",
                        language=detected_lang,
                        start_time=0.0,
                        end_time=audio.duration_seconds,
                        confidence=0.0,
                    )

                logger.info(f"Sherpa-ONNX Whisper recognized '{text}' (lang={detected_lang})")
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
            language=effective_lang or "es",
            start_time=0.0,
            end_time=audio.duration_seconds,
            confidence=0.0,
        )
