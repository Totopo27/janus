import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from janus.ports.llm_port import ILLMProvider
from janus.adapters.llm.mock_llm_adapter import MockLLMAdapter
from janus.adapters.llm.ollama_adapter import OllamaAdapter
from janus.adapters.llm.gemini_adapter import GeminiAdapter

load_dotenv()


class LLMProviderFactory:
    """
    Factory for instantiating the active BYOM LLM provider
    based on application configuration or dynamic user settings.
    """

    @staticmethod
    def create(provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> ILLMProvider:
        cfg = dict(config or {})
        cfg.update(kwargs)
        prov = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

        if prov == "mock":
            return MockLLMAdapter(
                default_response=cfg.get("default_response", "Respuesta simulada de Janus AI.")
            )

        if prov == "gemini":
            api_key = cfg.get("api_key") or os.getenv("GEMINI_API_KEY", "")
            model = cfg.get("model") or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
            return GeminiAdapter(api_key=api_key, model=model)

        # Default to local Ollama
        base_url = cfg.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = cfg.get("model") or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        return OllamaAdapter(base_url=base_url, model=model)

