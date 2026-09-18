from abc import ABC, abstractmethod
from janus.domain.models import TranslationResult


class ITranslator(ABC):
    """Port for Neural Machine Translation (MT)."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Translates text from source_lang to target_lang."""
        pass
