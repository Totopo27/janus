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

    # 2. Sherpa-ONNX Whisper Tiny / Small Multilingual
    sherpa_dir = os.path.join(target_dir, "sherpa-whisper")
    logger.info("Downloading Sherpa-ONNX Whisper models...")
    try:
        snapshot_download(
            repo_id="csukuangfj/sherpa-onnx-whisper-tiny",
            local_dir=sherpa_dir,
        )
        logger.info(f"Sherpa-ONNX models saved in: {sherpa_dir}")
    except Exception as e:
        logger.warning(f"Could not download Sherpa-ONNX models automatically: {e}")

    logger.info("Model download process completed.")


if __name__ == "__main__":
    download_models()
