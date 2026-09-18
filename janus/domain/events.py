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
