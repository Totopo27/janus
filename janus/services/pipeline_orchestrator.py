import logging
import uuid
from typing import Optional
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

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinates the Speech-to-Speech Translation (S2ST) pipeline:
    AudioChunk -> VAD -> STT -> MT -> TTS -> Broadcaster
    """

    def __init__(
        self,
        stt_engine: ISpeechRecognizer,
        translation_engine: ITranslator,
        tts_engine: ISpeechSynthesizer,
        broadcaster: Optional[IEventBroadcaster] = None,
        vad_engine: Optional[IVoiceActivityDetector] = None,
    ) -> None:
        self.stt = stt_engine
        self.mt = translation_engine
        self.tts = tts_engine
        self.broadcaster = broadcaster
        self.vad = vad_engine

    async def process_turn(
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
        transcription = self.stt.transcribe(audio=audio, language=source_lang)
        logger.info(f"[{session.session_id}] Transcribed ({source_lang}): '{transcription.text}'")

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
                    speaker_id=speaker_id,
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

        return turn
