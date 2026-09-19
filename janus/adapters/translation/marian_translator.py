import logging
import time
from typing import Optional
from janus.domain.models import TranslationResult
from janus.ports.translation_port import ITranslator

logger = logging.getLogger(__name__)


class MarianTranslator(ITranslator):
    """
    On-device, ultra-fast Machine Translation (MT) adapter.
    Uses MarianMT / Opus-MT neural models (or lightweight local translation models)
    with sub-50ms CPU execution.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_onnx: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_onnx = use_onnx
        self._pipeline = None

    def _init_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            if not self.model_name_or_path:
                raise RuntimeError("A local translation model path is required")
            logger.info("Loading configured local translation model...")
            self._pipeline = pipeline("translation", model=self.model_name_or_path)
        except Exception as e:
            logger.error("Translation model initialization failed (%s)", type(e).__name__)
            raise RuntimeError("Local translation model is not configured or could not be loaded") from None

    def translate(
        self,
        text: str,
        source_lang: str = "es",
        target_lang: str = "en",
    ) -> TranslationResult:
        self._init_pipeline()
        start = time.perf_counter()
        clean_text = text.strip()

        if not clean_text:
            return TranslationResult(
                source_text="",
                source_lang=source_lang,
                translated_text="",
                target_lang=target_lang,
                latency_ms=0.0,
            )

        try:
            res = self._pipeline(clean_text)
            translated = res[0]["translation_text"]
        except Exception as e:
            logger.error("Neural translation inference failed (%s)", type(e).__name__)
            raise RuntimeError("Translation inference failed") from None

        elapsed = (time.perf_counter() - start) * 1000.0
        return TranslationResult(
            source_text=clean_text,
            source_lang=source_lang,
            translated_text=translated,
            target_lang=target_lang,
            latency_ms=round(elapsed, 2),
        )
