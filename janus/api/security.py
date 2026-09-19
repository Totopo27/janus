from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass
from typing import Annotated, Iterable
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, WebSocket, status
from starlette.responses import JSONResponse


DEFAULT_TRUSTED_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
)
MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024


class RequestSizeLimitMiddleware:
    def __init__(self, app, max_body_bytes: int = MAX_HTTP_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                        scope, receive, send
                    )
                    return
            except (TypeError, ValueError):
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                scope, receive, send
            )


class _RequestBodyTooLarge(Exception):
    pass


def normalize_origin(origin: str) -> str:
    parsed = urlsplit(origin.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Trusted origins must be absolute HTTP(S) origins")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Trusted origins cannot contain credentials, query strings, or fragments")
    if parsed.path not in {"", "/"}:
        raise ValueError("Trusted origins cannot contain a path")

    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if parsed.scheme == "http" else 443
    port = parsed.port
    authority = host if port in {None, default_port} else f"{host}:{port}"
    return f"{parsed.scheme}://{authority}"


def _parse_origins(raw_origins: str | None) -> frozenset[str]:
    values: Iterable[str] = (
        raw_origins.split(",") if raw_origins else DEFAULT_TRUSTED_ORIGINS
    )
    origins = frozenset(normalize_origin(value) for value in values if value.strip())
    if not origins:
        raise RuntimeError("JANUS_TRUSTED_ORIGINS must contain at least one trusted origin")
    return origins


@dataclass(frozen=True)
class SecuritySettings:
    api_key: str
    admin_api_key: str
    trusted_origins: frozenset[str]
    websocket_auth_timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> "SecuritySettings":
        api_key = os.getenv("JANUS_API_KEY", "")
        admin_api_key = os.getenv("JANUS_ADMIN_API_KEY", "")
        if len(api_key) < 24 or len(admin_api_key) < 24:
            raise RuntimeError(
                "JANUS_API_KEY and JANUS_ADMIN_API_KEY must each be at least 24 characters"
            )
        if secrets.compare_digest(api_key, admin_api_key):
            raise RuntimeError("JANUS_API_KEY and JANUS_ADMIN_API_KEY must be different")
        return cls(
            api_key=api_key,
            admin_api_key=admin_api_key,
            trusted_origins=_parse_origins(os.getenv("JANUS_TRUSTED_ORIGINS")),
        )


class ApiAuthenticator:
    def __init__(self, settings: SecuritySettings) -> None:
        self.settings = settings

    @staticmethod
    def _bearer_token(authorization: str | None) -> str | None:
        if not authorization:
            return None
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and token:
            return token
        return None

    def _is_user_token(self, token: str | None) -> bool:
        if token is None:
            return False
        return secrets.compare_digest(token, self.settings.api_key) or secrets.compare_digest(
            token, self.settings.admin_api_key
        )

    def _is_admin_token(self, token: str | None) -> bool:
        return token is not None and secrets.compare_digest(token, self.settings.admin_api_key)

    def require_user(
        self,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not self._is_user_token(self._bearer_token(authorization)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def require_admin(
        self,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not self._is_admin_token(self._bearer_token(authorization)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")

    def is_trusted_origin(self, origin: str | None) -> bool:
        if not origin:
            return False
        try:
            normalized = normalize_origin(origin)
        except (ValueError, TypeError):
            return False
        return normalized in self.settings.trusted_origins

    async def authenticate_websocket(self, websocket: WebSocket) -> bool:
        if not self.is_trusted_origin(websocket.headers.get("origin")):
            await websocket.close(code=1008, reason="Untrusted WebSocket origin")
            return False

        await websocket.accept()
        try:
            payload = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=self.settings.websocket_auth_timeout_seconds,
            )
        except Exception:
            await websocket.close(code=1008, reason="WebSocket authentication required")
            return False

        token = payload.get("token") if isinstance(payload, dict) and payload.get("type") == "auth" else None
        if not isinstance(token, str) or not self._is_user_token(token):
            await websocket.close(code=1008, reason="Invalid WebSocket credentials")
            return False
        return True
