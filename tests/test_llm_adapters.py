import json
import httpx
from unittest.mock import patch, MagicMock
import pytest
from janus.domain.models import ChatMessage
from janus.ports.llm_port import ILLMProvider
from janus.adapters.llm.mock_llm_adapter import MockLLMAdapter
from janus.adapters.llm.ollama_adapter import OllamaAdapter
from janus.adapters.llm.gemini_adapter import GeminiAdapter
from janus.adapters.llm.factory import LLMProviderFactory


def test_mock_llm_adapter():
    adapter = MockLLMAdapter(default_response="Respuesta simulada")
    assert adapter.provider_name == "mock"
    assert adapter.health_check() is True

    resp = adapter.generate("¿Cuál es el estado del proyecto?")
    assert resp == "Respuesta simulada"

    chat_resp = adapter.chat_with_meeting(
        meeting_context="Transcripción de la reunión...",
        history=[ChatMessage(role="user", content="Hola")],
        question="¿Quién participó?",
    )
    assert "simulada" in chat_resp


def test_ollama_adapter_generate():
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="qwen2.5:3b")
    assert adapter.provider_name == "ollama"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "Janus es una plataforma local de traducción en tiempo real."
    }

    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = adapter.generate(
            prompt="Explica qué es Janus",
            system_prompt="Eres un asistente experto.",
        )
        assert "Janus" in result
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "http://localhost:11434/api/generate" in args[0]
        assert kwargs["json"]["model"] == "qwen2.5:3b"
        assert kwargs["json"]["system"] == "Eres un asistente experto."


def test_ollama_adapter_chat_with_meeting():
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="qwen2.5:3b")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Carlos propuso usar SQLite FTS5."}
    }

    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = adapter.chat_with_meeting(
            meeting_context="Carlos: Debemos usar SQLite FTS5.",
            history=[ChatMessage(role="user", content="¿Quién propuso la base de datos?")],
            question="¿Quién propuso la base de datos?",
        )
        assert "Carlos" in result
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "http://localhost:11434/api/chat" in args[0]
        messages = kwargs["json"]["messages"]
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "Carlos: Debemos usar SQLite FTS5." in messages[0]["content"]


def test_gemini_adapter_generate():
    adapter = GeminiAdapter(api_key="test_api_key", model="gemini-2.5-flash")
    assert adapter.provider_name == "gemini"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Respuesta desde Google Gemini Cloud"}]
                }
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = adapter.generate("Hola Gemini")
        assert result == "Respuesta desde Google Gemini Cloud"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "generativelanguage.googleapis.com" in args[0]
        assert "test_api_key" not in args[0]
        assert kwargs["headers"]["x-goog-api-key"] == "test_api_key"
        assert kwargs["follow_redirects"] is False


def test_gemini_adapter_chat_with_meeting():
    adapter = GeminiAdapter(api_key="test_api_key", model="gemini-2.5-flash")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "La reunión concluyó a las 15:00."}]
                }
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = adapter.chat_with_meeting(
            meeting_context="Reunión finalizada a las 15:00.",
            history=[],
            question="¿A qué hora terminó?",
        )
        assert "15:00" in result
        mock_post.assert_called_once()


def test_gemini_adapter_requires_api_key():
    adapter = GeminiAdapter(api_key="", model="gemini-2.5-flash")
    with pytest.raises(ValueError, match="API key"):
        adapter.generate("Test sin key")


def test_llm_factory():
    factory = LLMProviderFactory()

    # Create mock
    mock_prov = factory.create("mock")
    assert isinstance(mock_prov, MockLLMAdapter)

    # Create ollama
    ollama_prov = factory.create("ollama", {"base_url": "http://localhost:11434", "model": "qwen2.5:3b"})
    assert isinstance(ollama_prov, OllamaAdapter)

    # Create gemini
    gemini_prov = factory.create("gemini", {"api_key": "dummy_key", "model": "gemini-2.5-flash"})
    assert isinstance(gemini_prov, GeminiAdapter)


def test_ollama_rejects_private_and_metadata_networks():
    with pytest.raises(ValueError, match="Private|metadata"):
        OllamaAdapter(base_url="https://169.254.169.254", model="qwen2.5:3b")
    with pytest.raises(ValueError, match="HTTPS"):
        OllamaAdapter(base_url="http://10.0.0.10:11434", model="qwen2.5:3b")


def test_factory_ignores_request_level_base_url_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    provider = LLMProviderFactory.create(
        "ollama",
        {"base_url": "http://169.254.169.254", "model": "qwen2.5:3b"},
    )
    assert provider.base_url == "http://localhost:11434"


def test_gemini_errors_do_not_expose_api_keys():
    adapter = GeminiAdapter(api_key="top-secret-api-key", model="gemini-2.5-flash")
    request = MagicMock()
    response = MagicMock(status_code=500)
    error = httpx.HTTPStatusError("secret URL", request=request, response=response)
    with patch("httpx.post", side_effect=error):
        with pytest.raises(RuntimeError) as raised:
            adapter.generate("hello")
    assert "top-secret-api-key" not in str(raised.value)
