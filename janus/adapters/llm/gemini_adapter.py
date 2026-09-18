import logging
from typing import List, Optional
import httpx
from janus.domain.models import ChatMessage
from janus.ports.llm_port import ILLMProvider

logger = logging.getLogger(__name__)


class GeminiAdapter(ILLMProvider):
    """
    Cloud-based LLM adapter connecting to Google Gemini REST API.
    Provides scalable multimodal and high-context reasoning for meeting intelligence.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "gemini"

    def health_check(self) -> bool:
        if not self.api_key or not self.api_key.strip():
            return False
        try:
            url = f"{self.BASE_URL}?key={self.api_key}"
            resp = httpx.get(url, timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Gemini health check failed: %s", e)
            return False

    def _ensure_api_key(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Google Gemini API key is required to use the Gemini provider.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._ensure_api_key()
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts)
        except Exception as e:
            logger.error("Gemini generate request failed: %s", e)
            raise RuntimeError(f"Error communicating with Google Gemini: {e}") from e

    def chat_with_meeting(
        self,
        meeting_context: str,
        history: List[ChatMessage],
        question: str,
    ) -> str:
        self._ensure_api_key()
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        system_instruction = (
            "Eres el asistente inteligente de Janus para esta reunión. "
            "Responde a las preguntas de los participantes basándote en el contexto y transcripción provista. "
            "Si algo no está en la transcripción, indícalo con claridad.\n\n"
            f"--- CONTEXTO DE LA REUNIÓN ---\n{meeting_context}"
        )

        contents = []
        for msg in history:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

        contents.append({
            "role": "user",
            "parts": [{"text": question}],
        })

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": contents,
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts)
        except Exception as e:
            logger.error("Gemini chat request failed: %s", e)
            raise RuntimeError(f"Error communicating with Google Gemini chat: {e}") from e
