from janus.adapters.stt.sherpa_stt_adapter import SherpaSttAdapter, sanitize_transcription
from janus.domain.models import AudioChunk


def test_sanitizer_drops_existing_noise_labels():
    for text in ["[music]", "[MUSICA]", "[applause]", "(music)", "(musica)", ""]:
        assert sanitize_transcription(text, audio_duration_seconds=2.0) == ""


def test_sanitizer_drops_stock_video_closings_as_whole_lines():
    for text in [
        "¡Gracias por ver el video!",
        "Suscríbete al canal.",
        "Thanks for watching!",
        "Please subscribe.",
        "ご視聴ありがとうございました",
        "Продолжение следует.",
    ]:
        assert sanitize_transcription(text, audio_duration_seconds=2.0) == ""


def test_sanitizer_drops_multilingual_subtitle_credits():
    for text in [
        "Subtitles by Jane Doe",
        "Subtítulos realizados por la comunidad de Amara.org",
        "Untertitel von Stephanie Geiges",
        "Sous-titres réalisés par Jean Dupont",
        "Sottotitoli creati dalla comunità Amara.org",
        "Legendas pela comunidade Amara.org",
        "Napisy wykonane przez Jana Kowalskiego",
        "字幕由Amara.org社区提供",
        "Редактор субтитров А.Семкин",
    ]:
        assert sanitize_transcription(text, audio_duration_seconds=2.0) == ""


def test_sanitizer_keeps_legitimate_sentences_that_mention_filtered_terms():
    for text in [
        "Gracias por ver el presupuesto antes de aprobarlo.",
        "Gracias.",
        "Thank you.",
        "Bye.",
        "Who made the subtitles for the launch video?",
        "The subtitles by default are disabled.",
        "Music",
        "La música del evento comienza mañana.",
    ]:
        assert sanitize_transcription(text, audio_duration_seconds=2.0) == text


def test_sanitizer_drops_pathological_word_or_phrase_repetition_in_short_audio():
    assert sanitize_transcription("hola hola hola hola", audio_duration_seconds=2.0) == ""
    assert sanitize_transcription(
        "gracias por ver gracias por ver gracias por ver gracias por ver",
        audio_duration_seconds=5.0,
    ) == ""


def test_sanitizer_keeps_three_repetitions_or_repetition_in_long_audio():
    assert sanitize_transcription("hola hola hola", audio_duration_seconds=2.0) == "hola hola hola"
    repeated = "prueba de sonido prueba de sonido prueba de sonido prueba de sonido"
    assert sanitize_transcription(repeated, audio_duration_seconds=7.0) == repeated


def test_sanitizer_preserves_normal_text_unchanged():
    text = "Acordamos revisar el presupuesto el viernes a las 10:30."
    assert sanitize_transcription(text, audio_duration_seconds=3.0) == text


def test_sherpa_adapter_returns_an_empty_result_for_a_hallucination():
    class FakeResult:
        text = "Gracias por ver el video."
        lang = "es"

    class FakeStream:
        result = FakeResult()

        def accept_waveform(self, sample_rate, samples):
            assert sample_rate == 16000
            assert len(samples) == 32000

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def decode_stream(self, stream):
            return None

    adapter = SherpaSttAdapter(default_language="es")
    adapter._recognizers["es"] = FakeRecognizer()
    audio = AudioChunk(data=b"\x01\x00" * 32000, sample_rate=16000)

    result = adapter.transcribe(audio)

    assert result.text == ""
    assert result.language == "es"
    assert result.end_time == 2.0
    assert result.confidence == 0.0
