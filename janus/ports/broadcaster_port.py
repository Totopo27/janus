from abc import ABC, abstractmethod
from typing import Any, Dict
from janus.domain.events import DomainEvent


class IEventBroadcaster(ABC):
    """Port for broadcasting live events and subtitles to connected clients (teleprompter)."""

    @abstractmethod
    async def broadcast_event(self, session_id: str, event: DomainEvent) -> None:
        """Broadcasts a domain event to all clients subscribed to the session."""
        pass

    @abstractmethod
    async def broadcast_raw(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Broadcasts a raw dictionary payload to session subscribers."""
        pass
