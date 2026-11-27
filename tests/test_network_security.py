from unittest.mock import MagicMock, patch
import pytest
from janus.adapters.llm.network_security import validate_llm_base_url
from janus.adapters.llm.ollama_adapter import OllamaAdapter


def _mock_addrinfo(ip_str: str, port: int = 443):
    return [(None, None, None, None, (ip_str, port))]


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


@pytest.mark.parametrize("mapped_ip", ["::ffff:169.254.169.254", "::ffff:100.100.100.200"])
def test_validate_llm_base_url_rejects_ipv4_mapped_ipv6_metadata(mapped_ip):
    url = "https://metadata.example.com/v1"
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo(mapped_ip)):
        with pytest.raises(ValueError, match="Cloud metadata endpoints are not allowed"):
            validate_llm_base_url(url)


@pytest.mark.parametrize(
    "ip_addr",
    [
        "100.64.0.1",
        "10.0.0.1",
        "192.168.1.10",
        "172.16.0.1",
        "fc00::1",
        "fe80::1",
        "::ffff:10.0.0.1",
    ],
)
def test_validate_llm_base_url_rejects_cgnat_and_private_ranges(ip_addr):
    url = "https://internal.example.com/v1"
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo(ip_addr)):
        with pytest.raises(ValueError, match="Private and non-routable"):
            validate_llm_base_url(url)


@pytest.mark.parametrize("loopback_ip", ["127.0.0.1", "::1"])
def test_validate_llm_base_url_rejects_dns_to_loopback(loopback_ip):
    url = "https://rebind.example/v1"
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo(loopback_ip)):
        with pytest.raises(ValueError, match="loopback"):
            validate_llm_base_url(url)


def test_validate_llm_base_url_rejects_localhost_resolving_to_private():
    url = "http://localhost:11434"
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo("10.0.0.5", 11434)):
        with pytest.raises(ValueError, match="Private and non-routable"):
            validate_llm_base_url(url)


def test_ollama_adapter_revalidates_dns_and_disables_redirects():
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo("93.184.216.34")):
        adapter = OllamaAdapter(base_url="https://llm.example.com", model="qwen2.5:3b")

    # DNS rebinding after initialization must be blocked before httpx.post runs
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo("169.254.169.254")), \
         patch("httpx.post") as mock_post:
        with pytest.raises(ValueError, match="Cloud metadata endpoints are not allowed"):
            adapter.generate("hi")
        mock_post.assert_not_called()

    # Valid call passes follow_redirects=False
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "ok"}
    with patch("socket.getaddrinfo", return_value=_mock_addrinfo("93.184.216.34")), \
         patch("httpx.post", return_value=mock_resp) as mock_post:
        assert adapter.generate("hi") == "ok"
        assert mock_post.call_args.kwargs.get("follow_redirects") is False


