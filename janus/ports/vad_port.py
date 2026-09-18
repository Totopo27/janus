from abc import ABC, abstractmethod
from janus.domain.models import AudioChunk


class IVoiceActivityDetector(ABC):
    """Port for Voice Activity Detection (VAD)."""

    @abstractmethod
    def contains_speech(self, chunk: AudioChunk) -> bool:
        """Determines whether the given audio chunk contains human speech."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal VAD state."""
        pass
