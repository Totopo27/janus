import json
import logging
from typing import List, Optional
import urllib.request
import urllib.error
from janus.domain.models import ChatMessage
from janus.ports.llm_port import ILLMProvider

logger = logging.getLogger(__name__)


class ClaudeAdapter(ILLMProvider):
    """
    Anthropic Claude Cloud LLM Provider adapter using Messages API.
    Provides deep contextual synthesis for long multi-party interviews.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "claude"

    def health_check(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: Claude API Key no configurada."

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        body = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content_list = data.get("content", [])
                text_parts = [p.get("text", "") for p in content_list if p.get("type") == "text"]
                return "\n".join(text_parts).strip()
        except Exception as e:
            logger.error(f"Claude API request failed: {e}", exc_info=True)
            return f"Error en Claude API: {e}"

    def chat_with_meeting(
        self,
        meeting_context: str,
        history: List[ChatMessage],
        question: str,
    ) -> str:
        system = (
            "Eres el asistente inteligente de Janus para entrevistas y reuniones. "
            "Responde a las preguntas del usuario basándote estrictamente en el contexto de la conversación provisto.\n\n"
            f"<contexto_entrevista>\n{meeting_context}\n</contexto_entrevista>"
        )
        return self.generate(prompt=question, system_prompt=system)
