from dataclasses import dataclass, field
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from janus.domain.models import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class NemotronSpeakerSegment:
    start: float
    end: float
    speaker_index: int
    confidence: float = 1.0


@dataclass
class NemotronDiarizationReport:
    num_speakers: int
    is_monologue: bool
    has_overlap: bool
    segments: List[NemotronSpeakerSegment] = field(default_factory=list)
    overlap_duration: float = 0.0
    active_speakers: List[int] = field(default_factory=list)
    acoustic_hint: str = ""


class NemotronDiarizationService:
    """
    On-device, offline/streaming Speaker Diarization based on NVIDIA's Nemotron 3 (Sortformer) ONNX model.
    Detects up to 8 simultaneous speakers and resolves overlapping speech directly at the frame level.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.35,
        min_speech_duration: float = 0.25,
        subsampling_factor: int = 8,
        sample_rate: int = 16000,
    ) -> None:
        self.model_path = model_path
        self.probability_threshold = probability_threshold
        self.min_speech_duration = min_speech_duration
        self.subsampling_factor = subsampling_factor
        self.sample_rate = sample_rate

        self._session = None
        self._is_fallback = False

    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(project_root, "assets", "models", "diarization", "nemotron", "model_q4f16.onnx"),
            os.path.join(project_root, "assets", "models", "diarization", "nemotron", "model_quantized.onnx"),
            os.path.join(project_root, "assets", "models", "diarization", "nemotron", "model_fp16.onnx"),
            os.path.join(project_root, "assets", "models", "diarization", "nemotron", "model.onnx"),
        ]

        for cand in candidates:
            if os.path.exists(cand):
                return cand

        env_path = os.environ.get("NEMOTRON_DIARIZATION_MODEL")
        if env_path and os.path.exists(env_path):
            return env_path

        return None

    def _init_session(self) -> None:
        if self._session is not None or self._is_fallback:
            return

        resolved = self._resolve_model_path()
        if not resolved:
            logger.info("Nemotron 3 Diarization ONNX model not found. Operating in fallback mode.")
            self._is_fallback = True
            return

        try:
            import onnxruntime as ort

            # Prefer CUDA/DirectML if available, else CPU
            available = ort.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(resolved, sess_options=sess_options, providers=providers)
            logger.info(f"Nemotron 3 Diarization ONNX initialized from '{os.path.basename(resolved)}' with {providers[0]}")
        except Exception as exc:
            logger.warning(f"Failed to initialize Nemotron 3 Diarization ONNX: {exc}")
            self._is_fallback = True

    def compute_fbank(
        self,
        signal: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        preemphasis: float = 0.97,
    ) -> np.ndarray:
        """
        Extracts 128-dim log-mel filterbank features with 10ms frame hop and 25ms window,
        padding frames to a multiple of subsampling_factor (8) for the Transformer encoder.
        """
        if len(signal) == 0:
            return np.zeros((0, n_mels), dtype=np.float32)

        # Pre-emphasis filter
        emphasized = np.append(signal[0], signal[1:] - preemphasis * signal[:-1])

        # Framing
        if len(emphasized) < win_length:
            pad_amount = win_length - len(emphasized)
            emphasized = np.pad(emphasized, (0, pad_amount), mode="constant")

        num_frames = 1 + int(np.floor((len(emphasized) - win_length) / hop_length))
        rem = num_frames % self.subsampling_factor
        if rem != 0:
            pad_frames = self.subsampling_factor - rem
            target_len = (num_frames + pad_frames - 1) * hop_length + win_length
            if target_len > len(emphasized):
                emphasized = np.pad(emphasized, (0, target_len - len(emphasized)), mode="constant")
            num_frames += pad_frames

        window = np.hanning(win_length)
        frames = np.lib.stride_tricks.as_strided(
            emphasized,
            shape=(num_frames, win_length),
            strides=(emphasized.strides[0] * hop_length, emphasized.strides[0]),
        )
        windowed = frames * window

        # Power spectrum
        dft = np.fft.rfft(windowed, n=n_fft)
        power = (np.abs(dft) ** 2) / n_fft

        # Triangular mel filterbank
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)

        filters = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                filters[m - 1, k] = (k - bin_points[m - 1]) / max(bin_points[m] - bin_points[m - 1], 1e-8)
            for k in range(f_m, f_m_plus):
                filters[m - 1, k] = (bin_points[m + 1] - k) / max(bin_points[m + 1] - bin_points[m], 1e-8)

        mel_spec = np.dot(power, filters.T)
        mel_spec = np.where(mel_spec <= 0, np.finfo(float).eps, mel_spec)
        log_mel = np.log(mel_spec)
        return log_mel.astype(np.float32)

    def analyze(self, audio: AudioChunk) -> NemotronDiarizationReport:
        """
        Analyzes an AudioChunk using Nemotron 3 Diarization ONNX.
        Detects active speakers, continuous segments, and overlapping speech zones.
        """
        self._init_session()

        if self._is_fallback or self._session is None or audio.is_empty:
            return NemotronDiarizationReport(
                num_speakers=1,
                is_monologue=True,
                has_overlap=False,
                segments=[],
                acoustic_hint="Nemotron offline/fallback (single speaker default).",
            )

        try:
            samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
            mel = self.compute_fbank(samples)
            num_mel_frames = mel.shape[0]

            if num_mel_frames < self.subsampling_factor:
                return NemotronDiarizationReport(
                    num_speakers=1,
                    is_monologue=True,
                    has_overlap=False,
                    segments=[],
                    acoustic_hint="Audio too short for Nemotron diarization.",
                )

            num_encoder_frames = num_mel_frames // self.subsampling_factor
            input_features = np.expand_dims(mel, axis=0)
            cached_embeds = np.zeros((1, 0, 512), dtype=np.float32)
            attention_mask = np.ones((1, num_encoder_frames), dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_features": input_features,
                    "cached_embeds": cached_embeds,
                    "attention_mask": attention_mask,
                },
            )
            logits = outputs[0][0]  # shape: [num_mel_frames, 8]
            probs = 1.0 / (1.0 + np.exp(-logits))

            frame_duration = 0.01  # 10ms frame stride
            total_duration = num_mel_frames * frame_duration

            # Identify active speakers and extract segments per speaker
            active_speaker_set = set()
            segments: List[NemotronSpeakerSegment] = []

            for spk_idx in range(8):
                spk_probs = probs[:, spk_idx]
                is_active = spk_probs >= self.probability_threshold

                in_segment = False
                seg_start = 0.0

                for t, active in enumerate(is_active):
                    if active and not in_segment:
                        in_segment = True
                        seg_start = t * frame_duration
                    elif not active and in_segment:
                        in_segment = False
                        seg_end = t * frame_duration
                        if (seg_end - seg_start) >= self.min_speech_duration:
                            conf = float(np.mean(spk_probs[int(seg_start / frame_duration) : t]))
                            segments.append(
                                NemotronSpeakerSegment(
                                    start=round(seg_start, 2),
                                    end=round(seg_end, 2),
                                    speaker_index=spk_idx + 1,
                                    confidence=round(conf, 2),
                                )
                            )
                            active_speaker_set.add(spk_idx + 1)

                if in_segment:
                    seg_end = total_duration
                    if (seg_end - seg_start) >= self.min_speech_duration:
                        conf = float(np.mean(spk_probs[int(seg_start / frame_duration) :]))
                        segments.append(
                            NemotronSpeakerSegment(
                                start=round(seg_start, 2),
                                end=round(seg_end, 2),
                                speaker_index=spk_idx + 1,
                                confidence=round(conf, 2),
                            )
                        )
                        active_speaker_set.add(spk_idx + 1)

            # Sort segments chronologically
            segments.sort(key=lambda s: s.start)

            # Overlapping speech detection: frames where >= 2 speakers exceed probability threshold
            simultaneous_speakers_per_frame = np.sum(probs >= self.probability_threshold, axis=1)
            overlap_frames = np.sum(simultaneous_speakers_per_frame >= 2)
            overlap_duration = round(float(overlap_frames * frame_duration), 2)
            has_overlap = overlap_duration > 0.15

            num_speakers = len(active_speaker_set) if active_speaker_set else 1
            is_monologue = num_speakers <= 1 and not has_overlap

            if has_overlap:
                hint = (
                    f"Nemotron 3 detected OVERLAPPING SPEECH between {num_speakers} speakers "
                    f"({overlap_duration}s of simultaneous speech)."
                )
            elif is_monologue:
                hint = "Nemotron 3 confirmed a single speaker (monologue). No speaker competition."
            else:
                hint = f"Nemotron 3 detected {num_speakers} distinct non-overlapping sequential speakers."

            return NemotronDiarizationReport(
                num_speakers=num_speakers,
                is_monologue=is_monologue,
                has_overlap=has_overlap,
                segments=segments,
                overlap_duration=overlap_duration,
                active_speakers=sorted(list(active_speaker_set)),
                acoustic_hint=hint,
            )

        except Exception as exc:
            logger.error(f"Error in Nemotron diarization inference: {exc}", exc_info=True)
            return NemotronDiarizationReport(
                num_speakers=1,
                is_monologue=True,
                has_overlap=False,
                segments=[],
                acoustic_hint="Nemotron inference error, falling back to default.",
            )
