"""TTS adapters for Janus."""
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.tts.supertonic_tts_adapter import SupertonicTtsAdapter

__all__ = ["MockSpeechSynthesizer", "SupertonicTtsAdapter"]
