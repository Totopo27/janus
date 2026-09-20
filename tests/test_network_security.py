import pytest
from janus.adapters.llm.network_security import validate_llm_base_url


def test_validate_llm_base_url_valid_localhost():
    url = "http://localhost:11434"
    assert validate_llm_base_url(url) == "http://localhost:11434"


def test_validate_llm_base_url_valid_https():
    url = "https://api.openai.com/v1"
    assert validate_llm_base_url(url) == "https://api.openai.com/v1"


def test_validate_llm_base_url_rejects_http_remote():
    url = "http://api.unsecure-llm.com/v1"
    with pytest.raises(ValueError, match="Remote LLM endpoints must use HTTPS"):
        validate_llm_base_url(url)


def test_validate_llm_base_url_rejects_metadata_ip():
    url = "http://169.254.169.254/latest/meta-data"
    with pytest.raises(ValueError):
        validate_llm_base_url(url)


def test_validate_llm_base_url_rejects_credentials():
    url = "https://user:pass@api.openai.com"
    with pytest.raises(ValueError, match="credentials"):
        validate_llm_base_url(url)
