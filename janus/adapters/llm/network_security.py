from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


METADATA_ADDRESSES = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("100.100.100.200"),
}


def validate_llm_base_url(base_url: str) -> str:
    """Validate an operator-configured LLM endpoint and reject SSRF-sensitive networks."""
    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM base URL cannot contain credentials, query strings, or fragments")
    if parsed.path not in {"", "/"}:
        raise ValueError("LLM base URL cannot contain a path")

    hostname = parsed.hostname.lower()
    explicit_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not explicit_loopback:
        raise ValueError("Remote LLM endpoints must use HTTPS")

    try:
        addresses = {
            ipaddress.ip_address(sockaddr[0])
            for _, _, _, _, sockaddr in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError) as exc:
        raise ValueError("LLM base URL hostname could not be resolved safely") from exc

    if not addresses:
        raise ValueError("LLM base URL hostname did not resolve")

    for address in addresses:
        if address in METADATA_ADDRESSES:
            raise ValueError("Cloud metadata endpoints are not allowed")
        if address.is_loopback:
            if explicit_loopback:
                continue
            raise ValueError("DNS names resolving to loopback addresses are not allowed")
        if (
            address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError("Private and non-routable LLM endpoints are not allowed")

    return normalized
