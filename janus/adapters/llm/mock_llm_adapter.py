from typing import List, Optional
from janus.domain.models import ChatMessage
from janus.ports.llm_port import ILLMProvider


class MockLLMAdapter(ILLMProvider):
    """Mock LLM adapter for deterministic unit testing and offline development."""

    def __init__(self, default_response: str = "Respuesta simulada de Janus AI.") -> None:
        self.default_response = default_response
        self._healthy = True

    @property
    def provider_name(self) -> str:
        return "mock"

    def health_check(self) -> bool:
        return self._healthy

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self.default_response

    def chat_with_meeting(
        self,
        meeting_context: str,
        history: List[ChatMessage],
        question: str,
    ) -> str:
        return f"{self.default_response} (Sobre: '{question}')"
