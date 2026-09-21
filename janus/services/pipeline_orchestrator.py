import logging
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
    AudioChunk -> Diarization -> VAD -> STT -> MT -> TTS -> Broadcaster -> Storage
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
        diarizer: Optional[Any] = None,
        segmenter: Optional[Any] = None,
    ) -> None:
        self.stt = stt_engine
        self.mt = translation_engine
        self.tts = tts_engine
        self.broadcaster = broadcaster
        self.vad = vad_engine
        self.storage = storage_repo
        self.live_notetaker = live_notetaker
        self.diarizer = diarizer
        self.segmenter = segmenter

    async def process_turn(
        self,
        session: Session,
        speaker_id: str,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> Optional[ConversationTurn]:
        """
        Executes turn processing. If audio contains multiple distinct turns
        separated by silence, slices them and processes each individually.
        """
        if audio.is_empty:
            logger.debug("Received empty audio chunk, skipping turn.")
            return None

        if self.segmenter and audio.duration_seconds >= 2.0:
            sub_chunks = self.segmenter.segment_audio(audio)
            if len(sub_chunks) > 1:
                logger.info(f"[{session.session_id}] Silero VAD sliced multi-speaker audio into {len(sub_chunks)} turns.")
                last_turn = None
                for sub in sub_chunks:
                    turn = await self._process_single_turn(session, speaker_id, sub, language)
                    if turn:
                        last_turn = turn
                return last_turn

        return await self._process_single_turn(session, speaker_id, audio, language)

    async def _process_single_turn(
        self,
        session: Session,
        speaker_id: str,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> Optional[ConversationTurn]:
        """
        Executes a single conversational turn from audio input to translated speech.
        """
        if audio.is_empty:
            return None

        # 1. Voice Activity Detection (optional filter)
        if self.vad and not self.vad.contains_speech(audio):
            logger.debug("VAD detected no speech in audio chunk.")
            return None

        # 2. Acoustic Speaker Diarization for single-microphone scenarios
        effective_speaker_id = speaker_id
        effective_speaker_name = None
        if self.diarizer and speaker_id in ["local", "speaker_1", "speaker_2", ""]:
            diar_id, diar_name, conf = self.diarizer.identify_speaker(
                audio=audio,
                session_id=session.session_id,
                fallback_speaker_id=speaker_id,
            )
            if diar_id:
                effective_speaker_id = diar_id
                effective_speaker_name = diar_name
                logger.info(f"[{session.session_id}] Acoustic Diarization resolved speaker: {effective_speaker_id} ({diar_name})")

        # Identify current speaker and counterpart flexibly
        if effective_speaker_id in ["speaker_2", "remote"]:
            current_speaker = session.speaker_b
            counterpart = session.speaker_a
        else:
            current_speaker = session.speaker_a
            counterpart = session.speaker_b

        if not effective_speaker_name:
            effective_speaker_name = current_speaker.name

        # 3. Automatic Speech Recognition (with optional explicit language conditioning)
        effective_lang = language if language not in [None, "auto", ""] else None
        transcription = self.stt.transcribe(audio=audio, language=effective_lang)
        if not transcription or not transcription.text.strip():
            logger.debug(f"[{session.session_id}] No speech recognized or silence in audio chunk.")
            return None

        # Automatically determine source and target languages based on detected speech
        source_lang = transcription.language or "es"
        if source_lang.startswith("es"):
            target_lang = "en"
        elif source_lang.startswith("en"):
            target_lang = "es"
        else:
            target_lang = "en"

        logger.info(f"[{session.session_id}] Transcribed ({source_lang}) for '{effective_speaker_id}': '{transcription.text}'")

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TranscriptionCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=effective_speaker_id,
                    text=transcription.text,
                    language=source_lang,
                    confidence=transcription.confidence,
                ),
            )

        # 3. Machine Translation (MT)
        translation = self.mt.translate(
            text=transcription.text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        logger.info(f"[{session.session_id}] Translated ({target_lang}): '{translation.translated_text}'")

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TranslationCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=effective_speaker_id,
                    source_text=translation.source_text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    latency_ms=translation.latency_ms,
                ),
            )

        # 4. Text-to-Speech Synthesis (TTS)
        synthesis = self.tts.synthesize(
            text=translation.translated_text,
            language=target_lang,
            voice_style=counterpart.preferred_voice_style,
        )
        logger.info(f"[{session.session_id}] Synthesized TTS audio ({len(synthesis.audio_bytes)} bytes)")

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                SynthesisCompletedEvent(
                    session_id=session.session_id,
                    speaker_id=effective_speaker_id,
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
            speaker_id=effective_speaker_id,
            original_transcription=transcription,
            translation=translation,
            synthesis=synthesis,
        )
        session.add_turn(turn)

        # 6. Auto-persist to SQLite if storage repository is configured
        if self.storage:
            try:
                self.storage.save_turn(session.session_id, turn)
            except Exception as e:
                logger.warning(f"Failed to auto-persist turn to storage repository: {e}")

        if self.broadcaster:
            await self.broadcaster.broadcast_event(
                session.session_id,
                TurnCompletedEvent(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    speaker_id=effective_speaker_id,
                    speaker_name=effective_speaker_name,
                    original_text=transcription.text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                ),
            )

        # 7. Notify Live Notetaker in background
        if self.live_notetaker:
            try:
                asyncio.get_running_loop().create_task(
                    asyncio.to_thread(self.live_notetaker.process_turn, session.session_id, turn)
                )
            except Exception as e:
                logger.debug(f"Live notetaker background dispatch skipped: {e}")

        return turn

