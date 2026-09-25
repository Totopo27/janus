# Janus

> **Local-First, On-Device Speech-to-Speech Translation (S2ST) & Live Annotation Teleprompter.**
> *Break language barriers with zero cloud dependencies, sub-second latency, and complete privacy.*

---

## Vision & Concept

Named after **Janus**, the ancient Roman deity of transitions, passages, and dual perspectives, **Janus** bridges real-time conversation between two people speaking different languages.

Unlike traditional cloud translation APIs that introduce network lag, recurring costs, and privacy vulnerabilities, **Janus runs 100% on local CPU hardware**:
1. **Fast On-Device Ears (STT)**: Powered by **Sherpa-ONNX** (Next-gen Kaldi / Whisper INT8) with Voice Activity Detection (VAD).
2. **Lightweight Neural Translation (MT)**: Fast local neural translation (MarianMT / Opus-MT ONNX or CTranslate2) operating in ~40–50 ms.
3. **Studio-Quality On-Device Voice (TTS)**: Powered by **Supertonic** (ONNX Runtime, 99M parameters, 44.1 kHz studio audio) generating natural speech in ~120 ms on CPU.
4. **Live Teleprompter & Annotations**: Instant WebSockets broadcast to secondary screens, tablets, or phones to display real-time bilingual transcripts and meeting notes.

---

## Hexagonal Architecture (Ports & Adapters)

Janus strictly adheres to **Hexagonal Architecture** principles, ensuring that the conversational domain logic is fully decoupled from underlying audio drivers, inference engines, and transport layers:

```
                           ┌──────────────────────────────────────┐
                           │             CORE DOMAIN              │
                           │  Session, Turn, AudioChunk,          │
                           │  Transcription, Translation          │
                           └──────────────────┬───────────────────┘
                                              │
       ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
       ▼                  ▼                   ▼                   ▼                  ▼
  [Port: VAD]        [Port: STT]         [Port: MT]          [Port: TTS]       [Port: Broadcast]
       │                  │                   │                   │                  │
       ▼                  ▼                   ▼                   ▼                  ▼
(Silero/Sherpa       (Sherpa-ONNX       (MarianMT ONNX      (Supertonic         (FastAPI
 VAD Adapter)       Whisper / Zipf)     / NLLB Adapter)      ONNX Adapter)      WebSockets)
```

---

## Directory Structure

```text
janus/
├── janus/
│   ├── domain/               # Core business models and domain events
│   │   ├── models.py         # Session, Turn, AudioChunk, Transcription, Translation
│   │   └── events.py         # Domain events (SpeechDetected, TurnCompleted, etc.)
│   ├── ports/                # Abstract interfaces (Contracts)
│   │   ├── vad_port.py       # IVoiceActivityDetector
│   │   ├── stt_port.py       # ISpeechRecognizer
│   │   ├── translation_port.py# ITranslator
│   │   ├── tts_port.py       # ISpeechSynthesizer
│   │   └── broadcaster_port.py# IEventBroadcaster
│   ├── adapters/             # Concrete implementations of ports
│   │   ├── stt/              # Sherpa-ONNX and Mock adapters
│   │   ├── translation/      # MarianMT and Mock translators
│   │   ├── tts/              # Supertonic ONNX and Mock synthesizers
│   │   └── transport/        # FastAPI WebSockets live broadcaster
│   ├── services/             # Orchestration & application workflows
│   │   ├── pipeline_orchestrator.py # S2ST pipeline coordinator
│   │   └── session_service.py # Interlocutor session management
│   └── api/                  # HTTP & WebSocket endpoints
│       ├── app.py            # FastAPI main application
│       ├── routes.py         # REST management endpoints
│       └── websockets.py     # Real-time audio and teleprompter channels
├── web/                      # Standalone teleprompter & dual-speaker Web UI
│   ├── index.html            # Responsive teleprompter interface
│   ├── app.js                # WebSocket & audio streaming client
│   └── style.css             # High-contrast live subtitle styling
├── tests/                    # Strict TDD test suite
│   ├── test_domain_models.py
│   ├── test_pipeline_orchestrator.py
│   └── test_websocket_broadcaster.py
├── scripts/
│   └── download_models.py    # Automated ONNX weight downloader
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool

### 2. Environment Setup
```bash
git clone https://github.com/Totopo27/janus.git
cd janus

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests (Strict TDD)
```bash
pytest
```

### 4. Configuration & Models (Optional)
```bash
# Copy environment configuration template
cp .env.example .env

# Optional: Download offline ONNX models (Sherpa Whisper Small, CAM++, Silero VAD)
python scripts/download_models.py
```

### 5. Start Janus Server
```bash
# Standard local desktop run:
python run.py

# Multi-device Wi-Fi access (for connecting a tablet or smartphone as a teleprompter):
python run.py --lan
```

Open `http://localhost:8000` on your desktop, or the LAN IP displayed in console on your tablet or smartphone.

---

## Acoustic Feedback & Half-Duplex Gating

To prevent the speaker from picking up the generated synthetic voice and creating an infinite acoustic loop, Janus implements:
- **Half-Duplex Mic Gating**: The microphone stream is automatically suppressed while the TTS synthesis is playing back.
- **Push-to-Talk Mode**: Optional explicit turn trigger for noisy environments.

---

## License

MIT License. Designed and crafted with love and solid engineering fundamentals.
