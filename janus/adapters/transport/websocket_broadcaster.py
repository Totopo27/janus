import asyncio
import json
import logging
from typing import Any, Dict, Set
from fastapi import WebSocket
from janus.domain.events import DomainEvent
from janus.ports.broadcaster_port import IEventBroadcaster

logger = logging.getLogger(__name__)


class WebSocketBroadcaster(IEventBroadcaster):
    """Broadcaster managing active WebSocket connections per session."""

    def __init__(self) -> None:
        # Maps session_id -> Set of active WebSockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Registers a new WebSocket connection to a session."""
        async with self._lock:
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
        dead_sockets = []

        for ws in sockets:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket in session {session_id}: {e}")
                dead_sockets.append(ws)

        if dead_sockets:
            async with self._lock:
                for ws in dead_sockets:
                    if session_id in self._connections:
                        self._connections[session_id].discard(ws)
