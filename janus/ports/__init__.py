"""Abstract Ports (Contracts) for the Janus Hexagonal Architecture."""
from janus.ports.vad_port import IVoiceActivityDetector
from janus.ports.stt_port import ISpeechRecognizer
from janus.ports.translation_port import ITranslator
from janus.ports.tts_port import ISpeechSynthesizer
from janus.ports.broadcaster_port import IEventBroadcaster

__all__ = [
    "IVoiceActivityDetector",
    "ISpeechRecognizer",
    "ITranslator",
    "ISpeechSynthesizer",
    "IEventBroadcaster",
]
