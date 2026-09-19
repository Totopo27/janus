import asyncio
import json
import logging
from typing import Any, Dict, Set
from fastapi import WebSocket
from janus.domain.events import DomainEvent
from janus.ports.broadcaster_port import IEventBroadcaster

logger = logging.getLogger(__name__)
SEND_TIMEOUT_SECONDS = 5.0
MAX_CONNECTIONS_PER_SESSION = 25
MAX_TOTAL_CONNECTIONS = 250


class WebSocketBroadcaster(IEventBroadcaster):
    """Broadcaster managing active WebSocket connections per session."""

    def __init__(self) -> None:
        # Maps session_id -> Set of active WebSockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Registers a new WebSocket connection to a session."""
        async with self._lock:
            total_connections = sum(len(connections) for connections in self._connections.values())
            session_connections = self._connections.get(session_id, set())
            if (
                len(session_connections) >= MAX_CONNECTIONS_PER_SESSION
                or total_connections >= MAX_TOTAL_CONNECTIONS
            ):
                raise ConnectionError("WebSocket connection limit reached")
            if session_id not in self._connections:
                self._connections[session_id] = set()
            self._connections[session_id].add(websocket)
            logger.info(f"WebSocket client connected to session '{session_id}'. Total: {len(self._connections[session_id])}")

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Removes a WebSocket connection from a session."""
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]
                logger.info(f"WebSocket client disconnected from session '{session_id}'.")

    async def broadcast_event(self, session_id: str, event: DomainEvent) -> None:
        """Serializes and sends a domain event to all subscribers of session_id."""
        data_dict = {}
        for k, v in event.__dict__.items():
            if k == "occurred_at":
                continue
            if hasattr(v, "isoformat"):
                data_dict[k] = v.isoformat()
            else:
                data_dict[k] = v

        payload = {
            "type": "event",
            "event_name": event.event_name,
            "data": data_dict,
            "timestamp": event.occurred_at.isoformat() if hasattr(event.occurred_at, "isoformat") else str(event.occurred_at),
        }
        await self.broadcast_raw(session_id, payload)

    async def broadcast_raw(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Sends raw JSON payload to all active session WebSockets."""
        async with self._lock:
            sockets = list(self._connections.get(session_id, []))

        if not sockets:
            return

        message = json.dumps(payload)
        async def send(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_text(message), timeout=SEND_TIMEOUT_SECONDS)
                return None
            except Exception as e:
                logger.warning(
                    "Failed to send to WebSocket in session %s (%s)",
                    session_id,
                    type(e).__name__,
                )
                return ws

        results = await asyncio.gather(*(send(ws) for ws in sockets))
        dead_sockets = [ws for ws in results if ws is not None]

        if dead_sockets:
            async with self._lock:
                for ws in dead_sockets:
                    if session_id in self._connections:
                        self._connections[session_id].discard(ws)
