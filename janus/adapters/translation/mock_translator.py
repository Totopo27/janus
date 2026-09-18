from janus.domain.models import TranslationResult
from janus.ports.translation_port import ITranslator


class MockTranslator(ITranslator):
    """Mock translation adapter for deterministic unit testing."""

    def __init__(self, translation_map: dict[str, str] | None = None):
        self.translation_map = translation_map or {
            "Hola mundo": "Hello world",
            "Buenas tardes": "Good afternoon",
            "¿Cómo estás?": "How are you?",
            "Hello world": "Hola mundo",
        }
        self.translate_called_count = 0

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        self.translate_called_count += 1
        translated = self.translation_map.get(text, f"[{target_lang}] {text}")
        return TranslationResult(
            source_text=text,
            source_lang=source_lang,
            translated_text=translated,
            target_lang=target_lang,
            latency_ms=15.0,
        )
