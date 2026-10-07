import numpy as np
import pytest
from unittest.mock import MagicMock
from janus.domain.models import AudioChunk
from janus.services.speaker_diarization_service import (
    SpeakerDiarizationService,
    DiarizationSegment,
)
from janus.services.nemotron_diarization_service import (
    NemotronDiarizationService,
    NemotronSpeakerSegment,
    NemotronDiarizationReport,
)
from janus.services.conversational_fusion_service import ConversationalFusionService


def test_gate2_anti_phantom_speaker_filter():
    """
    Validates Gate 2:
    Micro-segments under 1.8 seconds (which normally produce spurious ghost speakers)
    must be reassigned to the nearest continuous speaker and fused cleanly.
    """
    service = SpeakerDiarizationService()
    # Simulated sequence with a 0.5s glitch speaker_2 between speaker_1 utterances
    raw_segments = [
        DiarizationSegment(start=0.0, end=4.0, speaker_index=1, confidence=0.95),
        DiarizationSegment(start=4.0, end=4.5, speaker_index=2, confidence=0.50),  # Phantom segment (<1.8s)
        DiarizationSegment(start=4.5, end=9.0, speaker_index=1, confidence=0.92),
    ]

    filtered = service._filter_phantom_segments(raw_segments)
    # The phantom segment must be reassigned and merged with speaker 1
    assert len(filtered) == 1
    assert filtered[0].speaker_index == 1
    assert filtered[0].start == 0.0
    assert filtered[0].end == 9.0


def test_gate2_multi_speaker_and_overlap_resolution():
    """
    Validates Gate 2:
    Simulates a 3-speaker interview with overlap speech (interruption/cross-talk).
    ConversationalFusionService must cleanly attribute turns to all 3 speakers
    without dropping statements or misattributing questions/answers.
    """
    mock_llm = MagicMock()
    # Simulated response from Qwen/Nemotron-aware fusion
    mock_llm.generate.return_value = '''
    [
        {
            "speaker": "carlos",
            "original": "¿Cuál es la principal ventaja de este diseño arquitectónico?",
            "translated": "What is the main advantage of this architectural design?"
        },
        {
            "speaker": "dra_elena",
            "original": "La resiliencia y el desacoplamiento total de los puertos.",
            "translated": "Resilience and complete decoupling of the ports."
        },
        {
            "speaker": "ing_marcos",
            "original": "Totalmente de acuerdo con la doctora Elena.",
            "translated": "Totally agree with Dr. Elena."
        }
    ]
    '''
    fusion = ConversationalFusionService(llm_provider=mock_llm)

    available_speakers = {
        "carlos": "Carlos (Entrevistador)",
        "dra_elena": "Dra. Elena Ramos",
        "ing_marcos": "Ing. Marcos",
    }

    turns = fusion.fuse_and_translate(
        text="¿Cuál es la principal ventaja de este diseño arquitectónico? La resiliencia y el desacoplamiento total de los puertos. Totalmente de acuerdo con la doctora Elena.",
        primary_speaker_id="carlos",
        primary_speaker_name="Carlos (Entrevistador)",
        counterpart_speaker_id="dra_elena",
        counterpart_speaker_name="Dra. Elena Ramos",
        source_lang="es",
        target_lang="en",
        acoustic_hint="Nemotron 3 detected OVERLAPPING SPEECH between 3 speakers (0.45s overlap).",
        available_speakers=available_speakers,
    )

    assert len(turns) == 3
    assert turns[0].speaker_id == "carlos"
    assert turns[0].speaker_name == "Carlos (Entrevistador)"

    assert turns[1].speaker_id == "dra_elena"
    assert turns[1].speaker_name == "Dra. Elena Ramos"

    assert turns[2].speaker_id == "ing_marcos"
    assert turns[2].speaker_name == "Ing. Marcos"
