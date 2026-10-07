import asyncio
import base64
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from janus.api.app import create_app
from janus.domain.models import SpeakerProfile, Session, ConversationTurn, TranscriptionResult, TranslationResult
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer


@pytest.fixture
def test_setup(tmp_path):
    session_service = SessionService()
    broadcaster = WebSocketBroadcaster()
    db_path = str(tmp_path / "stream_test.db")
    meeting_repo = SqliteMeetingRepository(db_path=db_path)

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Hola, probando audio móvil en streaming", language="es"),
        translation_engine=MockTranslator({"Hola, probando audio móvil en streaming": "Hello, testing mobile audio streaming"}),
        tts_engine=MockSpeechSynthesizer(),
        broadcaster=broadcaster,
        storage_repo=meeting_repo,
    )

    app = create_app(
        session_service=session_service,
        orchestrator=orchestrator,
        broadcaster=broadcaster,
        meeting_repo=meeting_repo,
    )

    # Pre-create a session
    spk_a = SpeakerProfile(speaker_id="mobile_user", name="Entrevistador Móvil", native_language="es")
    spk_b = SpeakerProfile(speaker_id="interviewee", name="Entrevistado", native_language="es")
    session_service.create_session("sess_mobile_stream", spk_a, spk_b)

    return app, broadcaster


def test_gate1_mobile_continuous_audio_streaming(test_setup):
    """
    Validates Gate 1:
    Continuous audio chunks sent from a simulated mobile client via WebSocket
    must be ingested, transcoded, and broadcast back as turn_result and domain events
    without buffer overflow, packet loss, or unhandled exceptions.
    """
    app, broadcaster = test_setup
    client = TestClient(app)

    # 1. Connect Teleprompter/HUD listener (as the mobile view screen)
    with client.websocket_connect("/ws/live-notes/sess_mobile_stream") as teleprompter_ws:
        init_msg = teleprompter_ws.receive_json()
        assert init_msg["type"] == "history"
        assert init_msg["session_id"] == "sess_mobile_stream"

        # 2. Connect Audio Stream WebSocket (simulating Mobile Web Audio API)
        with client.websocket_connect("/ws/audio-stream/sess_mobile_stream/mobile_user") as audio_ws:
            # Simulate streaming 5 consecutive audio slices (e.g. 5 x 2-second speech turns)
            # 16kHz, 16-bit PCM: 32000 samples = 2 seconds = 64000 bytes
            dummy_pcm16 = np.zeros(32000, dtype=np.int16).tobytes()
            b64_audio = base64.b64encode(dummy_pcm16).decode("utf-8")

            for chunk_idx in range(5):
                packet = {
                    "audio_base64": b64_audio,
                    "mime_type": "audio/wav",
                    "language": "es",
                    "target_language": "en",
                }
                audio_ws.send_text(json.dumps(packet))

                # Audio socket receives turn_result
                audio_res = audio_ws.receive_json()
                assert audio_res["type"] == "turn_result"
                assert "Hola, probando audio móvil" in audio_res["original_text"]
                assert audio_res["speaker_id"] in ["mobile_user", "speaker_1"]

                # Teleprompter socket receives broadcast domain events
                # PipelineProgress -> TranscriptionCompleted -> PipelineProgress -> TurnCompleted
                received_turn = False
                while not received_turn:
                    event = teleprompter_ws.receive_json()
                    if event.get("event_name") == "TurnCompleted":
                        assert event["data"]["session_id"] == "sess_mobile_stream"
                        assert "Hola, probando audio móvil" in event["data"]["original_text"]
                        received_turn = True


def test_gate1_reconnect_resilience_and_history(test_setup):
    """
    Validates that a mobile device disconnecting and reconnecting
    recovers the full interview history cleanly from in-memory state.
    """
    app, _ = test_setup
    client = TestClient(app)

    # First connection, do 1 turn
    with client.websocket_connect("/ws/audio-stream/sess_mobile_stream/mobile_user") as audio_ws:
        dummy_pcm16 = np.zeros(16000, dtype=np.int16).tobytes()
        audio_ws.send_text(json.dumps({
            "audio_base64": base64.b64encode(dummy_pcm16).decode("utf-8"),
            "mime_type": "audio/wav",
            "language": "es",
        }))
        res = audio_ws.receive_json()
        assert res["type"] == "turn_result"

    # Reconnect to live notes
    with client.websocket_connect("/ws/live-notes/sess_mobile_stream") as teleprompter_ws:
        history_msg = teleprompter_ws.receive_json()
        assert history_msg["type"] == "history"
        assert len(history_msg["turns"]) >= 1
        assert "Hola, probando audio móvil" in history_msg["turns"][0]["original_text"]
