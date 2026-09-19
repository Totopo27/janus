import logging
from typing import Dict, List, Optional, Any
from janus.domain.models import Session, SpeakerProfile

logger = logging.getLogger(__name__)


class SessionService:
    """In-memory session registry and dialogue manager for active conversations."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_session(
        self,
        session_id: str,
        speaker_a: SpeakerProfile,
        speaker_b: SpeakerProfile,
    ) -> Session:
        """Creates a session, or returns an identical existing session without resetting it."""
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.speaker_a == speaker_a and existing.speaker_b == speaker_b:
                return existing
            raise ValueError(f"Session '{session_id}' already exists with different participants")

        session = Session(
            session_id=session_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        self._sessions[session_id] = session
        logger.info(f"Session '{session_id}' created ({speaker_a.name} [{speaker_a.native_language}] <-> {speaker_b.name} [{speaker_b.native_language}])")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieves a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Session]:
        """Returns all active sessions."""
        return list(self._sessions.values())

    def close_session(self, session_id: str) -> bool:
        """Closes and removes an active session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session '{session_id}' closed.")
            return True
        return False

    def export_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """Exports the full conversation history of a session as a formatted transcript."""
        session = self.get_session(session_id)
        if not session:
            return []

        transcript = []
        for turn in session.turns:
            speaker = session.get_speaker(turn.speaker_id)
            speaker_name = speaker.name if speaker else turn.speaker_id
            transcript.append({
                "turn_id": turn.turn_id,
                "speaker_id": turn.speaker_id,
                "speaker_name": speaker_name,
                "original_text": turn.original_transcription.text,
                "source_lang": turn.original_transcription.language,
                "translated_text": turn.translation.translated_text,
                "target_lang": turn.translation.target_lang,
                "timestamp": turn.created_at,
            })
        return transcript
