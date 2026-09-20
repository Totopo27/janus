from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass(frozen=True)
class AudioChunk:
    """Represents a chunk of raw PCM audio data."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    timestamp: float = field(default_factory=time.time)

    @property
    def is_empty(self) -> bool:
        return len(self.data) == 0

    @property
    def duration_seconds(self) -> float:
        if self.is_empty or self.sample_rate <= 0:
            return 0.0
        # Assuming 16-bit linear PCM (2 bytes per sample per channel)
        bytes_per_sample = 2 * self.channels
        total_samples = len(self.data) / bytes_per_sample
        return total_samples / self.sample_rate


@dataclass(frozen=True)
class TranscriptionResult:
    """Represents the transcribed text from an audio segment."""
    text: str
    language: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class TranslationResult:
    """Represents the machine translation of a transcribed text."""
    source_text: str
    source_lang: str
    translated_text: str
    target_lang: str
    latency_ms: float = 0.0


@dataclass(frozen=True)
class SynthesisResult:
    """Represents the synthesized audio produced by the TTS engine."""
    audio_bytes: bytes
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    format: str = "wav"
    voice_id: str = "default"


@dataclass
class ConversationTurn:
    """A single dialogue turn within a conversational session."""
    turn_id: str
    session_id: str
    speaker_id: str
    original_transcription: TranscriptionResult
    translation: TranslationResult
    synthesis: Optional[SynthesisResult] = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SpeakerProfile:
    """Identifies a conversation participant and their language."""
    speaker_id: str
    name: str
    native_language: str
    preferred_voice_style: str = "default"


@dataclass
class Session:
    """An active conversational session connecting two speakers."""
    session_id: str
    speaker_a: SpeakerProfile
    speaker_b: SpeakerProfile
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        if self.speaker_a.speaker_id == speaker_id:
            return self.speaker_a
        if self.speaker_b.speaker_id == speaker_id:
            return self.speaker_b
        return None

    def get_counterpart(self, speaker_id: str) -> Optional[SpeakerProfile]:
        if self.speaker_a.speaker_id == speaker_id:
            return self.speaker_b
        if self.speaker_b.speaker_id == speaker_id:
            return self.speaker_a
        return None


@dataclass
class ActionItem:
    """A concrete task or commitment extracted from a conversation."""
    assignee: str
    task: str
    completed: bool = False
    due_hint: Optional[str] = None


@dataclass
class MeetingSummary:
    """Executive summary, key discussion points, and commitments for a meeting."""
    executive_summary: str
    key_points: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)


@dataclass
class Meeting:
    """A persistent meeting record containing participants, dialogue turns, and executive summary."""
    meeting_id: str
    title: str
    speaker_a: SpeakerProfile
    speaker_b: SpeakerProfile
    topic_key: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: Optional[MeetingSummary] = None
    live_notes: Optional['LiveMeetingNotes'] = None
    status: str = "active"  # "active" | "completed"
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def finalize(self, summary: MeetingSummary) -> None:
        self.summary = summary
        self.status = "completed"
        self.ended_at = time.time()

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        if self.speaker_a.speaker_id == speaker_id:
            return self.speaker_a
        if self.speaker_b.speaker_id == speaker_id:
            return self.speaker_b
        return None


@dataclass
class SearchResult:
    """A search hit from full-text search across meetings and turns."""
    meeting_id: str
    turn_id: str
    speaker_id: str
    original_text: str
    translated_text: str
    snippet: str
    topic_key: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ChatMessage:
    """A single turn in an interactive chat session with an AI model."""
    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiveMeetingNotes:
    """Real-time structured meeting notes continuously updated by Janus."""
    meeting_id: str
    current_topic: str = "Inicio de la reunión"
    key_takeaways: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    last_processed_turn_index: int = 0
    updated_at: float = field(default_factory=time.time)



