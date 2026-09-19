import logging
import asyncio
import uuid
from typing import Optional, Any
from janus.domain.models import (
    AudioChunk,
    ConversationTurn,
    Session,
)
from janus.domain.events import (
    TranscriptionCompletedEvent,
    TranslationCompletedEvent,
    SynthesisCompletedEvent,
    TurnCompletedEvent,
)
from janus.ports.vad_port import IVoiceActivityDetector
from janus.ports.stt_port import ISpeechRecognizer
from janus.ports.translation_port import ITranslator
from janus.ports.tts_port import ISpeechSynthesizer
from janus.ports.broadcaster_port import IEventBroadcaster
from janus.ports.storage_port import IMeetingRepository

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinates the Speech-to-Speech Translation (S2ST) pipeline:
    AudioChunk -> VAD -> STT -> MT -> TTS -> Broadcaster -> Storage
    """

    def __init__(
        self,
        stt_engine: ISpeechRecognizer,
        translation_engine: ITranslator,
        tts_engine: ISpeechSynthesizer,
        broadcaster: Optional[IEventBroadcaster] = None,
        vad_engine: Optional[IVoiceActivityDetector] = None,
        storage_repo: Optional[IMeetingRepository] = None,
        live_notetaker: Optional[Any] = None,
        max_concurrent_turns: int = 1,
    ) -> None:
        self.stt = stt_engine
        self.mt = translation_engine
        self.tts = tts_engine
        self.broadcaster = broadcaster
        self.vad = vad_engine
        self.storage = storage_repo
        self.live_notetaker = live_notetaker
        self._processing_semaphore = asyncio.Semaphore(max(1, max_concurrent_turns))
        self._background_tasks: set[asyncio.Task] = set()


    async def process_turn(
        self,
        session: Session,
        speaker_id: str,
        audio: AudioChunk,
    ) -> Optional[ConversationTurn]:
        async with self._processing_semaphore:
            return await self._process_turn(session=session, speaker_id=speaker_id, audio=audio)

    async def _process_turn(
        self,
        session: Session,
        speaker_id: str,
        audio: AudioChunk,
    ) -> Optional[ConversationTurn]:
        """
        Executes a single conversational turn from audio input to translated speech.
        """
        if audio.is_empty:
            logger.debug("Received empty audio chunk, skipping turn.")
            return None

        # 1. Voice Activity Detection (optional filter)
        if self.vad and not self.vad.contains_speech(audio):
            logger.debug("VAD detected no speech in audio chunk.")
            return None

        # Identify current speaker and counterpart target language
        current_speaker = session.get_speaker(speaker_id)
        if not current_speaker:
            raise ValueError(f"Speaker '{speaker_id}' does not belong to session '{session.session_id}'.")

        counterpart = session.get_counterpart(speaker_id)
        if not counterpart:
            raise ValueError(f"No counterpart found for speaker '{speaker_id}'.")

        source_lang = current_speaker.native_language
        target_lang = counterpart.native_language

        # 2. Automatic Speech Recognition (STT)
        transcription = await asyncio.to_thread(self.stt.transcribe, audio, source_lang)
        logger.info("[%s] Transcription completed (%s)", session.session_id, source_lang)

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TranscriptionCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=speaker_id,
                    text=transcription.text,
                    language=transcription.language,
                    confidence=transcription.confidence,
                ),
            )

        if not transcription.text.strip():
            return None

        # 3. Machine Translation (MT)
        translation = await asyncio.to_thread(
            self.mt.translate,
            transcription.text,
            source_lang,
            target_lang,
        )
        logger.info("[%s] Translation completed (%s)", session.session_id, target_lang)

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TranslationCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=speaker_id,
                    source_text=translation.source_text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    latency_ms=translation.latency_ms,
                ),
            )

        # 4. Text-to-Speech Synthesis (TTS)
        synthesis = await asyncio.to_thread(
            self.tts.synthesize,
            translation.translated_text,
            target_lang,
            counterpart.preferred_voice_style,
        )
        logger.info(f"[{session.session_id}] Synthesized TTS audio ({len(synthesis.audio_bytes)} bytes)")

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                SynthesisCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=speaker_id,
                    audio_bytes_length=len(synthesis.audio_bytes),
                    duration_seconds=synthesis.duration_seconds,
                    sample_rate=synthesis.sample_rate,
                ),
            )

        # 5. Record Conversation Turn
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=session.session_id,
            speaker_id=speaker_id,
            original_transcription=transcription,
            translation=translation,
            synthesis=synthesis,
        )
        # 6. Persist before publishing the turn so storage failures cannot be hidden.
        if self.storage:
            await asyncio.to_thread(self.storage.save_turn, session.session_id, turn)
        session.add_turn(turn)

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TurnCompletedEvent(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    speaker_id=speaker_id,
                    original_text=transcription.text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                ),
            )

        # 7. Notify Live Notetaker (Zoom AI Companion) in background
        if self.live_notetaker:
            task = asyncio.create_task(
                self.live_notetaker.process_turn_async(session.session_id, turn)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_task_finished)

        return turn

    def _background_task_finished(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Live notetaker background task failed (%s)", type(exc).__name__)
