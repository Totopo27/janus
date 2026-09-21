import pytest
from unittest.mock import MagicMock
from janus.services.conversational_fusion_service import ConversationalFusionService, FusedTurn


def test_fusion_monologue():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '''
    [
        {
            "speaker": "speaker_1",
            "original": "Hola a todos, bienvenidos a la reunión de hoy.",
            "translated": "Hello everyone, welcome to today's meeting."
        }
    ]
    '''
    service = ConversationalFusionService(llm_provider=mock_llm)

    turns = service.fuse_and_translate(
        text="Hola a todos, bienvenidos a la reunión de hoy.",
        primary_speaker_id="speaker_1",
        primary_speaker_name="Hablante 1",
        counterpart_speaker_id="speaker_2",
        counterpart_speaker_name="Hablante 2",
    )

    assert len(turns) == 1
    assert turns[0].speaker_id == "speaker_1"
    assert turns[0].speaker_name == "Hablante 1"
    assert "Hola a todos" in turns[0].original_text
    assert "Hello everyone" in turns[0].translated_text


def test_fusion_dialogue_breakdown():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '''
    [
        {
            "speaker": "speaker_1",
            "original": "¿Vives cerca de aquí?",
            "translated": "Do you live near here?"
        },
        {
            "speaker": "speaker_2",
            "original": "En realidad vivo en Ferrol. ¿Y tú?",
            "translated": "Actually, I live in Ferrol. And you?"
        },
        {
            "speaker": "speaker_1",
            "original": "No, vivo cerca.",
            "translated": "No, I live nearby."
        }
    ]
    '''
    service = ConversationalFusionService(llm_provider=mock_llm)

    turns = service.fuse_and_translate(
        text="¿Vives cerca de aquí? En realidad vivo en Ferrol. ¿Y tú? No, vivo cerca.",
        primary_speaker_id="speaker_1",
        primary_speaker_name="Hablante 1",
        counterpart_speaker_id="speaker_2",
        counterpart_speaker_name="Hablante 2",
    )

    assert len(turns) == 3
    assert turns[0].speaker_id == "speaker_1"
    assert turns[1].speaker_id == "speaker_2"
    assert turns[2].speaker_id == "speaker_1"
    assert turns[1].translated_text == "Actually, I live in Ferrol. And you?"


def test_fusion_fallback_on_error():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("LLM network timeout")

    mock_translator = MagicMock()
    mock_translator.translate.return_value = MagicMock(translated_text="Translated fallback")

    service = ConversationalFusionService(llm_provider=mock_llm, fallback_translator=mock_translator)

    turns = service.fuse_and_translate(
        text="Texto de prueba",
        primary_speaker_id="speaker_1",
        primary_speaker_name="Hablante 1",
    )

    assert len(turns) == 1
    assert turns[0].speaker_id == "speaker_1"
    assert turns[0].original_text == "Texto de prueba"
    assert turns[0].translated_text == "Translated fallback"
