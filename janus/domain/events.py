from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class DomainEvent:
    """Base class for domain events."""
    event_name: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(frozen=True)
class SpeechDetectedEvent(DomainEvent):
    session_id: str = ""
    speaker_id: str = ""
    duration_seconds: float = 0.0
    event_name: str = "SpeechDetected"


@dataclass(frozen=True)
class TranscriptionCompletedEvent(DomainEvent):
    session_id: str = ""
    speaker_id: str = ""
    text: str = ""
    language: str = ""
    confidence: float = 1.0
    event_name: str = "TranscriptionCompleted"


@dataclass(frozen=True)
class TranslationCompletedEvent(DomainEvent):
    session_id: str = ""
    speaker_id: str = ""
    source_text: str = ""
    translated_text: str = ""
    source_lang: str = ""
    target_lang: str = ""
    latency_ms: float = 0.0
    event_name: str = "TranslationCompleted"


@dataclass(frozen=True)
class SynthesisCompletedEvent(DomainEvent):
    session_id: str = ""
    speaker_id: str = ""
    audio_bytes_length: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    event_name: str = "SynthesisCompleted"


@dataclass(frozen=True)
class TurnCompletedEvent(DomainEvent):
    session_id: str = ""
    turn_id: str = ""
    speaker_id: str = ""
    original_text: str = ""
    translated_text: str = ""
    source_lang: str = ""
    target_lang: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_name: str = "TurnCompleted"


@dataclass(frozen=True)
class LiveNotesUpdatedEvent(DomainEvent):
    meeting_id: str = ""
    current_topic: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    event_name: str = "LiveNotesUpdated"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "meeting_id": self.meeting_id,
            "current_topic": self.current_topic,
            "key_takeaways": self.key_takeaways,
            "action_items": self.action_items,
        })
        return d

