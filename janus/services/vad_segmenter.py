import logging
import os
from typing import List, Optional
import numpy as np
from janus.domain.models import AudioChunk

logger = logging.getLogger(__name__)


class SileroVadSegmenter:
    """
    Server-side acoustic speech segmentation service powered by Sherpa-ONNX Silero VAD.
    Detects natural silence boundaries inside audio chunks and splits them into distinct
    speech turns to enable multi-speaker diarization and transcription within single recordings.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_silence_duration: float = 0.55,
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 25.0,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self._vad = None
        self._is_fallback = False

    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path

        env_path = os.environ.get("SILERO_VAD_MODEL")
        if env_path and os.path.exists(env_path):
            return env_path

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_dirs = [
            os.path.join(project_root, "assets", "models", "vad"),
            os.path.join(project_root, "assets", "models"),
        ]

        for cdir in candidate_dirs:
            if os.path.exists(cdir):
                for candidate in ["silero_vad.onnx", "silero_vad_v5.onnx"]:
                    p = os.path.join(cdir, candidate)
                    if os.path.exists(p):
                        return p
        return None

    def _init_vad(self) -> None:
        if self._vad is not None or self._is_fallback:
            return

        try:
            import sherpa_onnx

            model_file = self._resolve_model_path()
            if not model_file:
                logger.warning("No Silero VAD ONNX model found. Segmenter running in bypass mode.")
                self._is_fallback = True
                return

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = model_file
            config.silero_vad.threshold = self.threshold
            config.silero_vad.min_silence_duration = self.min_silence_duration
            config.silero_vad.min_speech_duration = self.min_speech_duration
            config.silero_vad.max_speech_duration = self.max_speech_duration
            config.sample_rate = 16000

            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
            logger.info(f"Silero VAD segmenter initialized from {os.path.basename(model_file)}")
        except Exception as e:
            logger.warning(f"Failed to initialize Sherpa-ONNX Silero VAD: {e}. Running in bypass mode.")
            self._is_fallback = True

    def segment_audio(self, audio: AudioChunk) -> List[AudioChunk]:
        """
        Analyzes audio chunk with Silero VAD. If multiple distinct speech segments
        separated by silence are detected, returns a list of individual AudioChunks.
        Otherwise returns [audio].
        """
        if audio.is_empty or audio.duration_seconds < 1.0:
            return [audio]

        self._init_vad()
        if self._is_fallback or self._vad is None:
            return [audio]

        try:
            self._vad.reset()
            samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0

            window_size = 512
            detected_segments = []

            for i in range(0, len(samples), window_size):
                chunk_slice = samples[i : i + window_size]
                self._vad.accept_waveform(chunk_slice)
                while not self._vad.empty():
                    seg = self._vad.front
                    if len(seg.samples) >= int(self.min_speech_duration * audio.sample_rate):
                        detected_segments.append(np.array(seg.samples, dtype=np.float32))
                    self._vad.pop()

            self._vad.flush()
            while not self._vad.empty():
                seg = self._vad.front
                if len(seg.samples) >= int(self.min_speech_duration * audio.sample_rate):
                    detected_segments.append(np.array(seg.samples, dtype=np.float32))
                self._vad.pop()

            if len(detected_segments) > 1:
                logger.info(f"Silero VAD split multi-speaker audio ({audio.duration_seconds:.2f}s) into {len(detected_segments)} distinct speech turns.")
                result_chunks = []
                for idx, seg_samples in enumerate(detected_segments):
                    pcm_bytes = (np.clip(seg_samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    result_chunks.append(
                        AudioChunk(
                            data=pcm_bytes,
                            sample_rate=audio.sample_rate,
                            channels=1,
                            timestamp=audio.timestamp + (idx * 0.1),
                        )
                    )
                return result_chunks

        except Exception as exc:
            logger.error(f"Silero VAD segmentation error: {exc}. Retaining original audio chunk.")

        return [audio]
