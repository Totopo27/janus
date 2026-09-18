from abc import ABC, abstractmethod
from typing import List, Optional
from janus.domain.models import ChatMessage


class ILLMProvider(ABC):
    """Port for Bring Your Own Model (BYOM) inference engines (e.g. Ollama, Gemini)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'ollama', 'gemini', 'mock')."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verifies if the LLM provider is reachable and operational."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates raw text response for a given prompt."""
        pass

    @abstractmethod
    def chat_with_meeting(
        self,
        meeting_context: str,
        history: List[ChatMessage],
        question: str,
    ) -> str:
        """Conducts a multi-turn Q&A grounded on meeting transcript context."""
        pass
