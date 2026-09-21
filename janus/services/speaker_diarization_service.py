from dataclasses import dataclass, field
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from janus.domain.models import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker_index: int
    confidence: float = 1.0


@dataclass
class DiarizationReport:
    num_speakers: int
    is_monologue: bool
    segments: List[DiarizationSegment] = field(default_factory=list)
    acoustic_hint: str = ""


class SpeakerDiarizationService:
    """
    On-device, offline Speaker Diarization and Voice Biometrics Service.
    Uses Sherpa-ONNX CAM++ (3D-Speaker) for acoustic embeddings and PyAnnote 3.0 ONNX
    for offline speaker segmentation and turn detection in single-microphone scenarios.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        segmentation_model_path: Optional[str] = None,
        num_threads: int = 2,
        similarity_threshold: float = 0.65,
        min_duration_seconds: float = 0.4,
    ) -> None:
        self.model_path = model_path
        self.segmentation_model_path = segmentation_model_path
        self.num_threads = num_threads
        self.similarity_threshold = similarity_threshold
        self.min_duration_seconds = min_duration_seconds

        self._extractor = None
        self._is_fallback = False
        self._diarizer = None
        self._is_diarizer_fallback = False
        self._session_managers: Dict[str, any] = {}
        self._session_speaker_names: Dict[str, Dict[str, str]] = {}

    def _init_extractor(self) -> None:
        if self._extractor is not None or self._is_fallback:
            return

        try:
            import sherpa_onnx

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate_dirs = [
                os.path.join(project_root, "assets", "models", "diarization"),
                os.path.join(project_root, "assets", "models"),
            ]

            resolved_path = self.model_path or os.environ.get("SPEAKER_EMBEDDING_MODEL")
            if not resolved_path or not os.path.exists(resolved_path):
                for candidate_dir in candidate_dirs:
                    if os.path.exists(candidate_dir):
                        for f in os.listdir(candidate_dir):
                            if f.endswith(".onnx") and ("campplus" in f.lower() or "speaker" in f.lower()):
                                resolved_path = os.path.join(candidate_dir, f)
                                break
                    if resolved_path and os.path.exists(resolved_path):
                        break

            if resolved_path and os.path.exists(resolved_path):
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=resolved_path,
                    num_threads=self.num_threads,
                    debug=False,
                    provider="cpu",
                )
                if not config.validate():
                    logger.warning(f"Speaker embedding config invalid for {resolved_path}")
                    self._is_fallback = True
                    return

                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
                logger.info(
                    f"Sherpa-ONNX SpeakerEmbeddingExtractor initialized successfully from "
                    f"{os.path.basename(resolved_path)} (dim={self._extractor.dim})."
                )
            else:
                logger.warning(
                    f"No CAM++ speaker embedding ONNX model found in {candidate_dirs}. "
                    "Speaker diarization operating in fallback mode."
                )
                self._is_fallback = True
        except Exception as exc:
            logger.warning(f"Could not initialize Sherpa-ONNX speaker diarization: {exc}")
            self._is_fallback = True

    def _init_diarizer(self) -> None:
        if self._diarizer is not None or self._is_diarizer_fallback:
            return

        try:
            import sherpa_onnx

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate_dirs = [
                os.path.join(project_root, "assets", "models", "diarization"),
                os.path.join(project_root, "assets", "models"),
            ]

            seg_path = self.segmentation_model_path or os.environ.get("SPEAKER_SEGMENTATION_MODEL")
            if not seg_path or not os.path.exists(seg_path):
                for candidate_dir in candidate_dirs:
                    if os.path.exists(candidate_dir):
                        for f in os.listdir(candidate_dir):
                            if f.endswith(".onnx") and ("pyannote" in f.lower() or "segmentation" in f.lower()):
                                seg_path = os.path.join(candidate_dir, f)
                                break
                    if seg_path and os.path.exists(seg_path):
                        break

            emb_path = self.model_path or os.environ.get("SPEAKER_EMBEDDING_MODEL")
            if not emb_path or not os.path.exists(emb_path):
                for candidate_dir in candidate_dirs:
                    if os.path.exists(candidate_dir):
                        for f in os.listdir(candidate_dir):
                            if f.endswith(".onnx") and ("campplus" in f.lower() or "speaker" in f.lower()):
                                emb_path = os.path.join(candidate_dir, f)
                                break
                    if emb_path and os.path.exists(emb_path):
                        break

            if seg_path and os.path.exists(seg_path) and emb_path and os.path.exists(emb_path):
                config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                            model=seg_path,
                        ),
                        num_threads=self.num_threads,
                        debug=False,
                        provider="cpu",
                    ),
                    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=emb_path,
                        num_threads=self.num_threads,
                        debug=False,
                        provider="cpu",
                    ),
                    clustering=sherpa_onnx.FastClusteringConfig(
                        num_clusters=2,
                        threshold=0.5,
                    ),
                    min_duration_on=0.3,
                    min_duration_off=0.5,
                )
                if not config.validate():
                    logger.warning("OfflineSpeakerDiarizationConfig validation failed")
                    self._is_diarizer_fallback = True
                    return

                self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
                logger.info(
                    f"Sherpa-ONNX PyAnnote OfflineSpeakerDiarization initialized successfully from {os.path.basename(seg_path)}."
                )
            else:
                logger.info("PyAnnote segmentation model not found; operating in acoustic fallback mode.")
                self._is_diarizer_fallback = True
        except Exception as exc:
            logger.warning(f"Could not initialize Sherpa-ONNX PyAnnote diarizer: {exc}")
            self._is_diarizer_fallback = True

    def _get_or_create_manager(self, session_id: str):
        import sherpa_onnx

        if session_id not in self._session_managers:
            self._session_managers[session_id] = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
            self._session_speaker_names[session_id] = {}
        return self._session_managers[session_id]

    def identify_speaker(
        self,
        audio: AudioChunk,
        session_id: str,
        fallback_speaker_id: str = "local",
        fallback_speaker_name: str = "Hablante Local",
    ) -> Tuple[str, str, float]:
        """
        Identifies or clusters the active speaker from an audio chunk using voice timbre embeddings.

        Returns:
            Tuple of (speaker_id, display_name, confidence_score)
        """
        self._init_extractor()

        if self._is_fallback or self._extractor is None or audio.is_empty:
            return fallback_speaker_id, fallback_speaker_name, 1.0

        if audio.duration_seconds < self.min_duration_seconds:
            # Too short to reliably compute biometric embedding; retain current/fallback
            return fallback_speaker_id, fallback_speaker_name, 0.5

        try:
            samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
            stream = self._extractor.create_stream()
            stream.accept_waveform(audio.sample_rate, samples)

            if not self._extractor.is_ready(stream):
                logger.debug(f"Audio stream not ready for embedding extraction ({len(samples)} samples)")
                return fallback_speaker_id, fallback_speaker_name, 0.5

            embedding = self._extractor.compute(stream)
            manager = self._get_or_create_manager(session_id)
            speaker_names_map = self._session_speaker_names[session_id]

            # Case 1: First speaker in this session
            if manager.num_speakers == 0:
                speaker_id = "speaker_1"
                display_name = "Hablante 1"
                manager.add(speaker_id, embedding)
                speaker_names_map[speaker_id] = display_name
                logger.info(f"[{session_id}] Registered first speaker voice profile '{speaker_id}' ({display_name})")
                return speaker_id, display_name, 1.0

            # Case 2: One speaker registered so far
            if manager.num_speakers == 1:
                registered_id = manager.all_speakers[0]
                similarity = manager.score(registered_id, embedding)
                logger.debug(f"[{session_id}] Similarity with '{registered_id}': {similarity:.3f} (thresh={self.similarity_threshold})")

                if similarity >= self.similarity_threshold:
                    # Matches speaker 1
                    return registered_id, speaker_names_map.get(registered_id, "Hablante 1"), float(similarity)
                else:
                    # New speaker detected! Register speaker 2
                    speaker_id = "speaker_2"
                    display_name = "Hablante 2"
                    manager.add(speaker_id, embedding)
                    speaker_names_map[speaker_id] = display_name
                    logger.info(
                        f"[{session_id}] Registered second speaker voice profile '{speaker_id}' "
                        f"({display_name}) [similarity to {registered_id}: {similarity:.3f}]"
                    )
                    return speaker_id, display_name, float(1.0 - similarity)

            # Case 3: Two or more speakers registered - find the best acoustic match
            best_id = manager.all_speakers[0] if manager.all_speakers else fallback_speaker_id
            best_score = -float("inf")

            for spk_id in manager.all_speakers:
                score = manager.score(spk_id, embedding)
                if score > best_score:
                    best_score = score
                    best_id = spk_id

            logger.info(f"[{session_id}] Best acoustic speaker match: '{best_id}' with score {best_score:.3f}")

            display_name = speaker_names_map.get(best_id, best_id.capitalize())
            return best_id, display_name, float(best_score)

        except Exception as exc:
            logger.error(f"Error in speaker identification: {exc}", exc_info=True)
            return fallback_speaker_id, fallback_speaker_name, 0.0

    def analyze_segments(self, audio: AudioChunk) -> DiarizationReport:
        """
        Analyzes audio chunk with PyAnnote 3.0 ONNX to detect speaker segments,
        providing an acoustic ground truth (monologue vs dialogue).
        """
        self._init_diarizer()

        if self._is_diarizer_fallback or self._diarizer is None or audio.is_empty or audio.duration_seconds < self.min_duration_seconds:
            return DiarizationReport(
                num_speakers=1,
                is_monologue=True,
                segments=[],
                acoustic_hint="monologue",
            )

        try:
            samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
            result = self._diarizer.process(samples)
            sorted_segs = result.sort_by_start_time()

            segments = [
                DiarizationSegment(
                    start=round(float(s.start), 2),
                    end=round(float(s.end), 2),
                    speaker_index=int(s.speaker),
                    confidence=round(float(s.confidence), 2) if hasattr(s, "confidence") else 1.0,
                )
                for s in sorted_segs
            ]

            unique_speakers = set(s.speaker_index for s in segments)
            num_speakers = len(unique_speakers) if unique_speakers else 1
            is_monologue = num_speakers <= 1

            if is_monologue:
                hint = "Acoustic diarizer confirmed a single speaker (monologue). Do NOT split into multiple speakers."
            else:
                switches = [f"{s.start}s (spk_{s.speaker_index})" for s in segments]
                hint = f"Acoustic diarizer detected {num_speakers} distinct speakers with segments at: {', '.join(switches)}."

            return DiarizationReport(
                num_speakers=num_speakers,
                is_monologue=is_monologue,
                segments=segments,
                acoustic_hint=hint,
            )
        except Exception as exc:
            logger.warning(f"Diarization segment analysis failed: {exc}")
            return DiarizationReport(
                num_speakers=1,
                is_monologue=True,
                segments=[],
                acoustic_hint="monologue",
            )

    def reset_session(self, session_id: str) -> None:
        """Clears registered speaker embeddings for a given session."""
        if session_id in self._session_managers:
            del self._session_managers[session_id]
        if session_id in self._session_speaker_names:
            del self._session_speaker_names[session_id]
        logger.info(f"Speaker embeddings reset for session '{session_id}'")
