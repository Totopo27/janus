import base64
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from janus.adapters.stt.audio_transcoding import transcode_to_pcm16
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
        logger.info(f"Audio stream client '{speaker_id}' connected to session '{session_id}'")
        session = session_service.get_session(session_id)

        if not session:
            logger.warning(f"Audio stream rejected: Session '{session_id}' not found for speaker '{speaker_id}'")
            await websocket.send_text(json.dumps({"error": "Session not found"}))
            await websocket.close()
            return

        try:
            while True:
                message = await websocket.receive()
                audio_bytes = b""
                mime_type = "audio/webm"

                if "bytes" in message and message["bytes"]:
                    audio_bytes = message["bytes"]
                    logger.info(f"[{session_id}:{speaker_id}] Received raw binary audio chunk ({len(audio_bytes)} bytes)")
                elif "text" in message and message["text"]:
                    try:
                        parsed = json.loads(message["text"])
                        if "audio_base64" in parsed:
                            audio_bytes = base64.b64decode(parsed["audio_base64"])
                            mime_type = parsed.get("mime_type", mime_type)
                            logger.info(f"[{session_id}:{speaker_id}] Received base64 audio chunk ({len(audio_bytes)} bytes decoded)")
                        else:
                            logger.warning(f"[{session_id}:{speaker_id}] JSON message missing 'audio_base64' key: {list(parsed.keys())}")
                    except Exception as pe:
                        logger.warning(f"[{session_id}:{speaker_id}] Could not parse JSON text message: {pe}")

                if not audio_bytes:
                    logger.warning(f"[{session_id}:{speaker_id}] Empty audio bytes received, skipping processing turn.")
                    continue

                pcm16_bytes = transcode_to_pcm16(audio_bytes, format_hint=mime_type)
                logger.info(f"[{session_id}:{speaker_id}] Transcoded audio ({len(audio_bytes)} bytes -> {len(pcm16_bytes)} pcm16 bytes). Dispatching to S2ST Pipeline Orchestrator...")
                chunk = AudioChunk(data=pcm16_bytes, sample_rate=16000)
                turn = await orchestrator.process_turn(
                    session=session,
                    speaker_id=speaker_id,
                    audio=chunk,
                )

                if turn:
                    logger.info(f"[{session_id}:{speaker_id}] Turn '{turn.turn_id}' completed: STT='{turn.original_transcription.text}' -> MT='{turn.translation.translated_text}'")
                    b64_audio = None
                    if turn.synthesis and turn.synthesis.audio_bytes:
                        b64_audio = base64.b64encode(turn.synthesis.audio_bytes).decode("utf-8")

                    payload = {
                        "type": "turn_result",
                        "turn_id": turn.turn_id,
                        "original_text": turn.original_transcription.text,
                        "translated_text": turn.translation.translated_text,
                    }
                    if b64_audio:
                        payload["audio_base64"] = b64_audio
                        payload["format"] = turn.synthesis.format
                        logger.info(f"[{session_id}:{speaker_id}] Dispatching turn_result with synthesized audio ({len(turn.synthesis.audio_bytes)} bytes)")
                    else:
                        logger.info(f"[{session_id}:{speaker_id}] Dispatching turn_result (text only, no audio in fallback mode)")

                    await websocket.send_text(json.dumps(payload))
                else:
                    logger.warning(f"[{session_id}:{speaker_id}] Pipeline orchestrator returned no turn (empty transcription or VAD filter)")
        except WebSocketDisconnect:
            logger.info(f"Audio stream client '{speaker_id}' disconnected from session '{session_id}'")
        except Exception as e:
            logger.error(f"Error in audio stream websocket for '{speaker_id}': {e}", exc_info=True)

    return router
