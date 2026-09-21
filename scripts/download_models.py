"""
Model Downloader Utility for Janus.
Downloads lightweight ONNX models for Sherpa-ONNX (Whisper/Zipformer) and Supertonic TTS.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_models(target_dir: str = "assets/models"):
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Target model directory: {os.path.abspath(target_dir)}")

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required. Install it via: pip install huggingface-hub")
        sys.exit(1)

    # 1. Supertonic 3 ONNX models
    supertonic_dir = os.path.join(target_dir, "supertonic")
    logger.info("Downloading Supertonic 3 ONNX assets...")
    try:
        snapshot_download(
            repo_id="supertone-oss-archive/supertonic-3",
            local_dir=supertonic_dir,
            ignore_patterns=["*.pt", "*.bin"],
        )
        logger.info(f"Supertonic assets saved in: {supertonic_dir}")
    except Exception as e:
        logger.warning(f"Could not download Supertonic automatically: {e}")

    # 2. Sherpa-ONNX Whisper Small / Base Multilingual
    whisper_dir = os.path.join(target_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    logger.info("Downloading Sherpa-ONNX Whisper models (Small INT8)...")
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt"]:
            target_path = os.path.join(whisper_dir, fname)
            if not os.path.exists(target_path):
                logger.info(f"Downloading {fname}...")
                hf_hub_download(
                    repo_id="csukuangfj/sherpa-onnx-whisper-small",
                    filename=fname,
                    local_dir=whisper_dir,
                )
        logger.info(f"Sherpa-ONNX Whisper Small models saved in: {whisper_dir}")
    except Exception as e:
        logger.warning(f"Could not download Sherpa-ONNX Whisper Small models automatically: {e}")

    # 3. Sherpa-ONNX Speaker Diarization / Embedding Models (CAM++)
    diarization_dir = os.path.join(target_dir, "diarization")
    os.makedirs(diarization_dir, exist_ok=True)
    logger.info("Downloading Speaker Embedding model (CAM++ 3D-Speaker)...")
    try:
        from huggingface_hub import hf_hub_download
        campplus_file = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
        target_path = os.path.join(diarization_dir, campplus_file)
        if not os.path.exists(target_path):
            logger.info(f"Downloading {campplus_file}...")
            hf_hub_download(
                repo_id="csukuangfj/speaker-embedding-models",
                filename=campplus_file,
                local_dir=diarization_dir,
            )
        logger.info(f"Speaker Diarization model saved in: {diarization_dir}")
    except Exception as e:
        logger.warning(f"Could not download Speaker Diarization model automatically: {e}")

    # 4. Sherpa-ONNX Voice Activity Detection (Silero VAD)
    vad_dir = os.path.join(target_dir, "vad")
    os.makedirs(vad_dir, exist_ok=True)
    logger.info("Downloading Silero VAD model...")
    try:
        from huggingface_hub import hf_hub_download
        vad_file = "silero_vad.onnx"
        target_path = os.path.join(vad_dir, vad_file)
        if not os.path.exists(target_path):
            logger.info(f"Downloading {vad_file}...")
            hf_hub_download(
                repo_id="csukuangfj/vad",
                filename=vad_file,
                local_dir=vad_dir,
            )
        logger.info(f"Silero VAD model saved in: {vad_dir}")
    except Exception as e:
        logger.warning(f"Could not download Silero VAD model automatically: {e}")

    logger.info("Model download process completed.")


if __name__ == "__main__":
    download_models()
