from unittest.mock import patch
import pytest
from janus.adapters.llm.network_security import validate_llm_base_url


def test_validate_llm_base_url_valid_localhost():
    url = "http://localhost:11434"
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 11434))]):
        assert validate_llm_base_url(url) == "http://localhost:11434"


def test_validate_llm_base_url_valid_https():
    url = "https://api.openai.com/v1"
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
        assert validate_llm_base_url(url) == "https://api.openai.com/v1"


def test_validate_llm_base_url_rejects_http_remote():
    url = "http://api.unsecure-llm.com/v1"
    with pytest.raises(ValueError, match="Remote LLM endpoints must use HTTPS"):
        validate_llm_base_url(url)


def test_validate_llm_base_url_rejects_metadata_ip():
    url = "https://169.254.169.254/latest/meta-data"
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("169.254.169.254", 443))]):
        with pytest.raises(ValueError, match="Cloud metadata endpoints are not allowed"):
            validate_llm_base_url(url)


def test_validate_llm_base_url_rejects_credentials():
    url = "https://user:pass@api.openai.com"
    with pytest.raises(ValueError, match="credentials"):
        validate_llm_base_url(url)

