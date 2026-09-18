from abc import ABC, abstractmethod
from typing import List, Optional
from janus.domain.models import Meeting, ConversationTurn, MeetingSummary, SearchResult


class IMeetingRepository(ABC):
    """Port for persistent storage of meetings, dialogue turns, and executive summaries."""

    @abstractmethod
    def save_meeting(self, meeting: Meeting) -> None:
        """Persists a new or updated meeting."""
        pass

    @abstractmethod
    def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Retrieves a meeting with all its turns and summary by ID."""
        pass

    @abstractmethod
    def list_meetings(self) -> List[Meeting]:
        """Lists all meetings ordered by creation date."""
        pass

    @abstractmethod
    def save_turn(self, meeting_id: str, turn: ConversationTurn) -> None:
        """Appends a new conversation turn to an existing meeting."""
        pass

    @abstractmethod
    def save_summary(self, meeting_id: str, summary: MeetingSummary) -> None:
        """Persists the meeting summary and sets status to completed."""
        pass

    @abstractmethod
    def delete_meeting(self, meeting_id: str) -> bool:
        """Removes a meeting record and its associated turns."""
        pass

    @abstractmethod
    def search_turns(self, query: str, topic_key: Optional[str] = None) -> List[SearchResult]:
        """Performs full-text search across dialogue turns with optional topic_key filtering."""
        pass

