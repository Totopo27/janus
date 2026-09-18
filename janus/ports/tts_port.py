from abc import ABC, abstractmethod
from typing import Optional
from janus.domain.models import SynthesisResult


class ISpeechSynthesizer(ABC):
    """Port for Text-to-Speech (TTS) synthesis."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        voice_style: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesizes text into speech audio."""
        pass
