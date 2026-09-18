from typing import Optional
from janus.domain.models import AudioChunk, TranscriptionResult
from janus.ports.stt_port import ISpeechRecognizer


class MockSpeechRecognizer(ISpeechRecognizer):
    """Mock STT adapter for deterministic unit testing."""

    def __init__(self, predefined_text: str = "Hola mundo", language: str = "es"):
        self.predefined_text = predefined_text
        self.language = language
        self.transcribe_called_count = 0

    def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        self.transcribe_called_count += 1
        lang = language or self.language
        return TranscriptionResult(
            text=self.predefined_text,
            language=lang,
            start_time=0.0,
            end_time=audio.duration_seconds,
            confidence=0.99,
        )
