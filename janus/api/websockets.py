import base64
import binascii
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from janus.domain.models import AudioChunk
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.session_service import SessionService
from janus.api.security import ApiAuthenticator
from janus.adapters.stt.audio_transcoding import AudioTranscodingError, transcode_to_pcm16

logger = logging.getLogger(__name__)
MAX_ENCODED_AUDIO_CHARS = 7_000_000
MAX_COMPRESSED_AUDIO_BYTES = 5 * 1024 * 1024
MAX_NOTE_MESSAGE_CHARS = 8_000
MAX_CONCURRENT_AUDIO_STREAMS = 8


def create_websocket_router(
    session_service: SessionService,
    orchestrator: PipelineOrchestrator,
    broadcaster: WebSocketBroadcaster,
    authenticator: ApiAuthenticator,
) -> APIRouter:
    router = APIRouter(tags=["WebSockets"])
    audio_connections: set[int] = set()
    audio_connections_lock = asyncio.Lock()

    @router.websocket("/ws/live-notes/{session_id}")
    async def websocket_live_notes(websocket: WebSocket, session_id: str):
        """
        WebSocket channel for the live teleprompter display.
        Streams real-time transcriptions, translations, and notes.
        """
        if not await authenticator.authenticate_websocket(websocket):
            return
        try:
            await broadcaster.connect(session_id, websocket)
        except ConnectionError:
            await websocket.close(code=1013, reason="Too many subscribers")
            return

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
                if len(data) > MAX_NOTE_MESSAGE_CHARS:
                    await websocket.close(code=1009, reason="Message too large")
                    return
                logger.debug("Received keepalive message for session %s", session_id)
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
        if not await authenticator.authenticate_websocket(websocket):
            return
        logger.info(f"Audio stream client '{speaker_id}' connected to session '{session_id}'")
        session = session_service.get_session(session_id)

        if not session:
            logger.warning(f"Audio stream rejected: Session '{session_id}' not found for speaker '{speaker_id}'")
            await websocket.send_text(json.dumps({"error": "Session not found"}))
            await websocket.close()
            return

        connection_id = id(websocket)
        async with audio_connections_lock:
            if len(audio_connections) >= MAX_CONCURRENT_AUDIO_STREAMS:
                await websocket.close(code=1013, reason="Too many audio streams")
                return
            audio_connections.add(connection_id)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                audio_bytes = b""
                audio_format = "audio/pcm;rate=16000"

                if "bytes" in message and message["bytes"]:
                    audio_bytes = message["bytes"]
                    logger.info(f"[{session_id}:{speaker_id}] Received raw binary audio chunk ({len(audio_bytes)} bytes)")
                elif "text" in message and message["text"]:
                    try:
                        parsed = json.loads(message["text"])
                        if not isinstance(parsed, dict):
                            raise ValueError("Audio message must be an object")
                        encoded_audio = parsed.get("audio_base64")
                        if isinstance(encoded_audio, str):
                            if len(encoded_audio) > MAX_ENCODED_AUDIO_CHARS:
                                await websocket.close(code=1009, reason="Audio payload too large")
                                return
                            audio_bytes = base64.b64decode(encoded_audio, validate=True)
                            audio_format = parsed.get("audio_format", "audio/webm")
                            if not isinstance(audio_format, str) or len(audio_format) > 80:
                                raise ValueError("Invalid audio content type")
                            logger.info(f"[{session_id}:{speaker_id}] Received base64 audio chunk ({len(audio_bytes)} bytes decoded)")
                        else:
                            raise ValueError("Audio message is missing audio_base64")
                    except (json.JSONDecodeError, binascii.Error, ValueError):
                        await websocket.send_text(json.dumps({"error": "Invalid audio message"}))
                        continue

                if not audio_bytes:
                    logger.warning(f"[{session_id}:{speaker_id}] Empty audio bytes received, skipping processing turn.")
                    continue
                if len(audio_bytes) > MAX_COMPRESSED_AUDIO_BYTES:
                    await websocket.close(code=1009, reason="Audio payload too large")
                    return

                logger.info(f"[{session_id}:{speaker_id}] Dispatching AudioChunk ({len(audio_bytes)} bytes) to S2ST Pipeline Orchestrator...")
                try:
                    pcm_bytes = await asyncio.to_thread(
                        transcode_to_pcm16,
                        audio_bytes,
                        audio_format,
                        16000,
                    )
                except AudioTranscodingError:
                    await websocket.send_text(json.dumps({"error": "Unsupported or invalid audio payload"}))
                    continue
                chunk = AudioChunk(data=pcm_bytes, sample_rate=16000)
                turn = await orchestrator.process_turn(
                    session=session,
                    speaker_id=speaker_id,
                    audio=chunk,
                )

                if turn:
                    logger.info("[%s:%s] Turn '%s' completed", session_id, speaker_id, turn.turn_id)
                    if turn.synthesis and turn.synthesis.audio_bytes:
                        b64_audio = base64.b64encode(turn.synthesis.audio_bytes).decode("utf-8")
                        await websocket.send_text(json.dumps({
                            "type": "turn_result",
                            "turn_id": turn.turn_id,
                            "original_text": turn.original_transcription.text,
                            "translated_text": turn.translation.translated_text,
                            "audio_base64": b64_audio,
                            "format": turn.synthesis.format,
                        }))
                        logger.info(f"[{session_id}:{speaker_id}] Dispatched turn_result to client with synthesized audio ({len(turn.synthesis.audio_bytes)} bytes)")
                else:
                    logger.warning(f"[{session_id}:{speaker_id}] Pipeline orchestrator returned no turn (empty transcription or VAD filter)")
        except WebSocketDisconnect:
            logger.info(f"Audio stream client '{speaker_id}' disconnected from session '{session_id}'")
        except Exception as e:
            logger.error(
                "Error in audio stream websocket for '%s' (%s)",
                speaker_id,
                type(e).__name__,
            )
        finally:
            async with audio_connections_lock:
                audio_connections.discard(connection_id)

    return router
