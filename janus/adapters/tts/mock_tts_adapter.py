from typing import Optional
from janus.domain.models import SynthesisResult
from janus.ports.tts_port import ISpeechSynthesizer


class MockSpeechSynthesizer(ISpeechSynthesizer):
    """Mock TTS adapter for deterministic unit testing."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.synthesize_called_count = 0

    def synthesize(
        self,
        text: str,
        language: str,
        voice_style: Optional[str] = None,
    ) -> SynthesisResult:
        self.synthesize_called_count += 1
        # Generar un buffer WAV mínimo simulado
        dummy_audio = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 1000
        return SynthesisResult(
            audio_bytes=dummy_audio,
            sample_rate=self.sample_rate,
            duration_seconds=1.0,
            format="wav",
            voice_id=voice_style or "default",
        )
