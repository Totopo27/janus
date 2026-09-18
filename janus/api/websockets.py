import base64
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from janus.domain.models import AudioChunk
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.session_service import SessionService

logger = logging.getLogger(__name__)


def create_websocket_router(
    session_service: SessionService,
    orchestrator: PipelineOrchestrator,
    broadcaster: WebSocketBroadcaster,
) -> APIRouter:
    router = APIRouter(tags=["WebSockets"])

    @router.websocket("/ws/live-notes/{session_id}")
    async def websocket_live_notes(websocket: WebSocket, session_id: str):
        """
        WebSocket channel for the live teleprompter display.
        Streams real-time transcriptions, translations, and notes.
        """
        await websocket.accept()
        await broadcaster.connect(session_id, websocket)

        try:
            # Send initial state/history
            history = session_service.export_transcript(session_id)
            await websocket.send_text(json.dumps({
                "type": "history",
                "session_id": session_id,
                "turns": history,
            }))

            # Keep connection alive while broadcaster pushes events
            while True:
                # Expect ping/keepalive or client notes
                data = await websocket.receive_text()
                # If client sends a manual note, echo or handle
                logger.debug(f"Received note message from teleprompter in {session_id}: {data}")
        except WebSocketDisconnect:
            logger.info(f"Teleprompter client disconnected from {session_id}")
        except Exception as e:
            logger.warning(f"Teleprompter websocket exception in {session_id}: {e}")
        finally:
            await broadcaster.disconnect(session_id, websocket)

    @router.websocket("/ws/audio-stream/{session_id}/{speaker_id}")
    async def websocket_audio_stream(
        websocket: WebSocket,
        session_id: str,
        speaker_id: str,
    ):
        """
        WebSocket channel for streaming client audio to the Janus S2ST pipeline.
        Can receive binary PCM/WAV chunks or JSON packets.
        """
        await websocket.accept()
        session = session_service.get_session(session_id)

        if not session:
            await websocket.send_text(json.dumps({"error": "Session not found"}))
            await websocket.close()
            return

        try:
            while True:
                message = await websocket.receive()
                audio_bytes = b""

                if "bytes" in message and message["bytes"]:
                    audio_bytes = message["bytes"]
                elif "text" in message and message["text"]:
                    try:
                        parsed = json.loads(message["text"])
                        if "audio_base64" in parsed:
                            audio_bytes = base64.b64decode(parsed["audio_base64"])
                    except Exception:
                        pass

                if not audio_bytes:
                    continue

                chunk = AudioChunk(data=audio_bytes, sample_rate=16000)
                turn = await orchestrator.process_turn(
                    session=session,
                    speaker_id=speaker_id,
                    audio=chunk,
                )

                if turn and turn.synthesis and turn.synthesis.audio_bytes:
                    # Send back the synthesized audio to the counterpart or caller
                    b64_audio = base64.b64encode(turn.synthesis.audio_bytes).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "turn_result",
                        "turn_id": turn.turn_id,
                        "original_text": turn.original_transcription.text,
                        "translated_text": turn.translation.translated_text,
                        "audio_base64": b64_audio,
                        "format": turn.synthesis.format,
                    }))
        except WebSocketDisconnect:
            logger.info(f"Audio stream client '{speaker_id}' disconnected from session '{session_id}'")
        except Exception as e:
            logger.error(f"Error in audio stream websocket: {e}")

    return router
