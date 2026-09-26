import asyncio
import logging
import uuid
from typing import Optional, Any
from janus.domain.models import (
    AudioChunk,
    ConversationTurn,
    Session,
    TranscriptionResult,
    TranslationResult,
    SynthesisResult,
)
from janus.services.conversational_fusion_service import FusedTurn
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
        fusion_service: Optional[Any] = None,
        enable_vad_slicing: bool = False,
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
        self.fusion_service = fusion_service
        self.enable_vad_slicing = enable_vad_slicing

    async def process_turn(
        self,
        session: Session,
        speaker_id: str,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> Optional[ConversationTurn]:
        """
        Executes turn processing for a conversational turn.
        Audio is processed as a complete, unbroken acoustic unit by default to ensure
        Whisper maintains full sentence context and avoids hallucination.
        """
        if audio.is_empty:
            logger.debug("Received empty audio chunk, skipping turn.")
            return None

        if self.enable_vad_slicing and self.segmenter and audio.duration_seconds >= 2.0:
            sub_chunks = await asyncio.to_thread(self.segmenter.segment_audio, audio)
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

        # 1. Voice Activity Detection (offloaded to thread to prevent loop blocking)
        if self.vad:
            has_speech = await asyncio.to_thread(self.vad.contains_speech, audio)
            if not has_speech:
                logger.debug("VAD detected no speech in audio chunk.")
                return None

        # 2. Acoustic Speaker Diarization (offloaded to thread)
        effective_speaker_id = speaker_id
        effective_speaker_name = None
        acoustic_hint = None

        if self.diarizer:
            # Nemotron 3 Diarization (Sortformer) with native Overlap Detection
            if hasattr(self.diarizer, "analyze") and not hasattr(self.diarizer, "analyze_segments"):
                try:
                    nemotron_report = await asyncio.to_thread(self.diarizer.analyze, audio)
                    acoustic_hint = nemotron_report.acoustic_hint
                    logger.info(
                        f"[{session.session_id}] Nemotron Diarization: {nemotron_report.num_speakers} speaker(s), "
                        f"monologue={nemotron_report.is_monologue}, overlap={nemotron_report.has_overlap} "
                        f"({nemotron_report.overlap_duration}s)"
                    )
                    # If Nemotron resolved a primary dominant speaker in the chunk
                    if nemotron_report.segments and speaker_id in ["local", "speaker_1", "speaker_2", ""]:
                        primary_idx = nemotron_report.segments[0].speaker_index
                        effective_speaker_id = f"speaker_{primary_idx}"
                        speaker_profile = session.get_speaker(effective_speaker_id)
                        if speaker_profile:
                            effective_speaker_name = speaker_profile.name
                except Exception as ne:
                    logger.debug(f"Nemotron segment analysis skipped: {ne}")

            # PyAnnote fallback if present
            elif hasattr(self.diarizer, "analyze_segments"):
                try:
                    report = await asyncio.to_thread(self.diarizer.analyze_segments, audio)
                    acoustic_hint = report.acoustic_hint
                    logger.info(f"[{session.session_id}] PyAnnote Diarization: {report.num_speakers} speaker(s), monologue={report.is_monologue}")
                except Exception as de:
                    logger.debug(f"Segment analysis skipped: {de}")

            if hasattr(self.diarizer, "identify_speaker") and speaker_id in ["local", "speaker_1", "speaker_2", ""]:
                diar_res = await asyncio.to_thread(
                    self.diarizer.identify_speaker,
                    audio=audio,
                    session_id=session.session_id,
                    fallback_speaker_id=speaker_id,
                )
                diar_id, diar_name, conf = diar_res
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
        transcription = await asyncio.to_thread(self.stt.transcribe, audio=audio, language=effective_lang)
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

        # 3. Conversational Fusion or Machine Translation
        if self.fusion_service:
            fused_turns = await asyncio.to_thread(
                self.fusion_service.fuse_and_translate,
                text=transcription.text,
                primary_speaker_id=effective_speaker_id,
                primary_speaker_name=effective_speaker_name,
                counterpart_speaker_id=counterpart.speaker_id,
                counterpart_speaker_name=counterpart.name,
                source_lang=source_lang,
                target_lang=target_lang,
                acoustic_hint=acoustic_hint,
            )
        else:
            translation = await asyncio.to_thread(
                self.mt.translate,
                text=transcription.text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            fused_turns = [
                FusedTurn(
                    speaker_id=effective_speaker_id,
                    speaker_name=effective_speaker_name,
                    original_text=transcription.text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            ]

        last_turn = None
        for idx, fused in enumerate(fused_turns):
            fused_turn_id = f"turn_{uuid.uuid4().hex[:8]}"
            fused_transcription = TranscriptionResult(
                text=fused.original_text,
                language=fused.source_lang,
                confidence=transcription.confidence,
            )
            fused_translation = TranslationResult(
                source_text=fused.original_text,
                source_lang=fused.source_lang,
                translated_text=fused.translated_text,
                target_lang=fused.target_lang,
                latency_ms=0.0,
            )

            # 4. Text-to-Speech Synthesis (TTS)
            synthesis = SynthesisResult(audio_bytes=b"", sample_rate=16000, duration_seconds=0.0, format="wav")
            try:
                synthesis = await asyncio.to_thread(
                    self.tts.synthesize,
                    text=fused.translated_text,
                    language=fused.target_lang,
                    voice_style=counterpart.preferred_voice_style,
                )
            except Exception as te:
                logger.debug(f"TTS synthesis skipped: {te}")

            # 5. Record Conversation Turn
            turn = ConversationTurn(
                turn_id=fused_turn_id,
                session_id=session.session_id,
                speaker_id=fused.speaker_id,
                original_transcription=fused_transcription,
                translation=fused_translation,
                synthesis=synthesis,
            )
            session.add_turn(turn)

            # 6. Auto-persist to SQLite
            if self.storage:
                try:
                    await asyncio.to_thread(self.storage.save_turn, session.session_id, turn)
                except Exception as e:
                    logger.warning(f"Failed to auto-persist turn to storage repository: {e}")

            # Broadcast TurnCompletedEvent to Teleprompter in real time
            if self.broadcaster:
                await self.broadcaster.broadcast_event(
                    session.session_id,
                    TurnCompletedEvent(
                        session_id=session.session_id,
                        turn_id=fused_turn_id,
                        speaker_id=fused.speaker_id,
                        speaker_name=fused.speaker_name,
                        original_text=fused.original_text,
                        translated_text=fused.translated_text,
                        source_lang=fused.source_lang,
                        target_lang=fused.target_lang,
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

            last_turn = turn

        return last_turn

