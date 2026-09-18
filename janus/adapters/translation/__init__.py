"""Translation adapters for Janus."""
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.translation.marian_translator import MarianTranslator

__all__ = ["MockTranslator", "MarianTranslator"]
