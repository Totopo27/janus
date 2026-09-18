from abc import ABC, abstractmethod
from typing import Optional
from janus.domain.models import AudioChunk, TranscriptionResult


class ISpeechRecognizer(ABC):
    """Port for Automatic Speech Recognition (ASR / STT)."""

    @abstractmethod
    def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribes an audio chunk into text."""
        pass
