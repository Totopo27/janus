import pytest

from janus.adapters.stt.sherpa_stt_adapter import SherpaSttAdapter
from janus.adapters.translation.marian_translator import MarianTranslator
from janus.adapters.tts.supertonic_tts_adapter import SupertonicTtsAdapter
from janus.domain.models import AudioChunk


def test_stt_without_models_fails_instead_of_fabricating_transcript():
    with pytest.raises(RuntimeError, match="not configured"):
        SherpaSttAdapter().transcribe(AudioChunk(b"\x00\x00" * 100))


def test_translation_without_model_fails_instead_of_echoing_source_text():
    with pytest.raises(RuntimeError, match="not configured"):
        MarianTranslator().translate("Hola", "es", "en")


def test_tts_without_model_fails_instead_of_generating_tone():
    with pytest.raises(RuntimeError, match="not configured"):
        SupertonicTtsAdapter(model_dir=None).synthesize("Hola", "es")
