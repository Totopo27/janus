import asyncio
import concurrent.futures
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import numpy as np
import pytest

from janus.adapters.llm.claude_adapter import ClaudeAdapter
from janus.adapters.llm.gemini_adapter import GeminiAdapter
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.marian_translator import MarianTranslator
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.adapters.tts.supertonic_tts_adapter import SupertonicTtsAdapter
from janus.domain.models import (
    AudioChunk,
    ChatMessage,
    ConversationTurn,
    Meeting,
    Session,
    SpeakerProfile,
    TranscriptionResult,
    TranslationResult,
)
from janus.ports.llm_port import ILLMProvider
from janus.services.live_notetaker_service import LiveNotetakerService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.speaker_diarization_service import SpeakerDiarizationService
from janus.services.vad_segmenter import SileroVadSegmenter


def test_vad_segmenter_serializes_concurrent_calls():
    segmenter = SileroVadSegmenter()
    state_lock = threading.Lock()
    state = {"active": 0, "max_concurrent": 0}

    class StubVad:
        def reset(self):
            pass

        def accept_waveform(self, _chunk_slice):
            with state_lock:
                state["active"] += 1
                state["max_concurrent"] = max(state["max_concurrent"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1

        def empty(self):
            return True

        def flush(self):
            pass

    segmenter._vad = StubVad()
    chunk = AudioChunk(data=b"\x00\x01" * 16000, sample_rate=16000)  # 1.0s

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(segmenter.segment_audio, chunk) for _ in range(4)]
        for fut in futures:
            fut.result()

    assert state["max_concurrent"] == 1


def test_speaker_diarization_clamps_confidence_and_serializes_access():
    service = SpeakerDiarizationService(similarity_threshold=0.65)
    state_lock = threading.Lock()
    state = {"active": 0, "max_concurrent": 0}

    class StubExtractor:
        dim = 8

        def create_stream(self):
            return SimpleNamespace(accept_waveform=lambda _sr, _samples: None)

        def is_ready(self, _stream):
            return True

        def compute(self, _stream):
            return np.ones(8, dtype=np.float32)

    class StubManager:
        def __init__(self):
            self._speakers = ["speaker_1"]

        @property
        def num_speakers(self):
            return len(self._speakers)

        @property
        def all_speakers(self):
            return list(self._speakers)

        def score(self, _speaker_id, _embedding):
            with state_lock:
                state["active"] += 1
                state["max_concurrent"] = max(state["max_concurrent"], state["active"])
            time.sleep(0.015)
            with state_lock:
                state["active"] -= 1
            return -0.35

        def add(self, speaker_id, _embedding):
            if speaker_id not in self._speakers:
                self._speakers.append(speaker_id)

    service._extractor = StubExtractor()
    service._session_managers["sess_conf"] = StubManager()
    service._session_speaker_names["sess_conf"] = {"speaker_1": "Hablante 1"}

    chunk = AudioChunk(data=b"\x00\x01" * 16000, sample_rate=16000)  # 1.0s
    spk_id, _, conf = service.identify_speaker(chunk, "sess_conf")
    assert spk_id == "speaker_2"
    assert 0.0 <= conf <= 1.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(service.identify_speaker, chunk, "sess_conf") for _ in range(4)]
        for fut in futures:
            _, _, fut_conf = fut.result()
            assert 0.0 <= fut_conf <= 1.0

    assert state["max_concurrent"] == 1


def test_live_notetaker_no_duplicate_llm_calls_or_lost_pending_turns(tmp_path):
    db_file = str(tmp_path / "live_race.db")
    repo = SqliteMeetingRepository(db_path=db_file)
    meeting = Meeting(
        meeting_id="meet_race",
        title="Race Test",
        speaker_a=SpeakerProfile("carlos", "Carlos", "es"),
        speaker_b=SpeakerProfile("alice", "Alice", "en"),
    )
    repo.save_meeting(meeting)

    turns = []
    for idx in range(4):
        t = ConversationTurn(
            turn_id=f"t_race_{idx}",
            session_id="meet_race",
            speaker_id="carlos" if idx % 2 == 0 else "alice",
            original_transcription=TranscriptionResult(f"Punto {idx}", "es"),
            translation=TranslationResult(f"Punto {idx}", "es", f"Point {idx}", "en"),
        )
        repo.save_turn("meet_race", t)
        turns.append(t)

    started = threading.Event()
    release = threading.Event()

    class BlockingLLM(ILLMProvider):
        def __init__(self):
            self.calls = 0
            self._call_lock = threading.Lock()

        @property
        def provider_name(self) -> str:
            return "blocking_mock"

        def health_check(self) -> bool:
            return True

        def generate(self, prompt: str, system_prompt=None) -> str:
            with self._call_lock:
                self.calls += 1
                call_num = self.calls
            if call_num == 1:
                started.set()
                release.wait(timeout=5.0)
            return json.dumps({
                "current_topic": f"Tema {call_num}",
                "key_takeaways": [f"Takeaway {call_num}"],
                "action_items": [],
            })

        def chat_with_meeting(self, meeting_context, history, question) -> str:
            return "ok"

    llm = BlockingLLM()
    service = LiveNotetakerService(repository=repo, llm_provider=llm, batch_size=2)

    def send_first_pair():
        service.process_turn("meet_race", turns[0])
        service.process_turn("meet_race", turns[1])

    worker = threading.Thread(target=send_first_pair)
    worker.start()

    assert started.wait(timeout=3.0)
    # Turn 3 arrives while call #1 is in-flight: must NOT trigger a second concurrent LLM call
    res3 = service.process_turn("meet_race", turns[2])
    assert res3 is None

    release.set()
    worker.join(timeout=3.0)

    assert llm.calls == 1
    assert service._pending_turns_count["meet_race"] == 1

    # Turn 4 completes the second batch and triggers call #2
    res4 = service.process_turn("meet_race", turns[3])
    assert res4 is not None
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_pipeline_orchestrator_speaker_lookup_custom_ids_and_third_speaker():
    speaker_a = SpeakerProfile("alice", "Alice", "en")
    speaker_b = SpeakerProfile("bob", "Bob", "es")
    session = Session(session_id="sess_custom", speaker_a=speaker_a, speaker_b=speaker_b)

    broadcaster = MagicMock()
    broadcaster.broadcast_event = AsyncMock()
    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Hola equipo", language="es"),
        translation_engine=MockTranslator({"Hola equipo": "Hello team"}),
        tts_engine=MockSpeechSynthesizer(),
        broadcaster=broadcaster,
    )
    chunk = AudioChunk(data=b"\x00\x01" * 1600, sample_rate=16000)

    turn_bob = await orchestrator.process_turn(session, "bob", chunk)
    assert turn_bob is not None
    assert turn_bob.speaker_id == "bob"
    assert turn_bob.translation.target_lang == "en"

    turn_events_bob = [
        c.args[1] for c in broadcaster.broadcast_event.await_args_list
        if c.args[1].event_name == "TurnCompleted"
    ]
    assert turn_events_bob[-1].speaker_name == "Bob"

    turn_3 = await orchestrator.process_turn(session, "speaker_3", chunk)
    assert turn_3 is not None
    assert turn_3.speaker_id == "speaker_3"

    turn_events_3 = [
        c.args[1] for c in broadcaster.broadcast_event.await_args_list
        if c.args[1].event_name == "TurnCompleted"
    ]
    assert turn_events_3[-1].speaker_name == "speaker_3"


@pytest.mark.asyncio
async def test_pipeline_orchestrator_logs_background_task_exception(caplog):
    speaker_a = SpeakerProfile("alice", "Alice", "es")
    speaker_b = SpeakerProfile("bob", "Bob", "en")
    session = Session(session_id="sess_bg_err", speaker_a=speaker_a, speaker_b=speaker_b)

    class FailingNotetaker:
        async def process_turn_async(self, _meeting_id, _turn):
            raise RuntimeError("boom_notetaker")

    orchestrator = PipelineOrchestrator(
        stt_engine=MockSpeechRecognizer(predefined_text="Hola", language="es"),
        translation_engine=MockTranslator({"Hola": "Hello"}),
        tts_engine=MockSpeechSynthesizer(),
        live_notetaker=FailingNotetaker(),
    )
    chunk = AudioChunk(data=b"\x00\x01" * 1600, sample_rate=16000)

    with caplog.at_level(logging.WARNING):
        await orchestrator.process_turn(session, "alice", chunk)
        await asyncio.sleep(0.05)

    assert len(orchestrator._background_tasks) == 0
    assert any(
        rec.name == "janus.services.pipeline_orchestrator" and "boom_notetaker" in rec.getMessage()
        for rec in caplog.records
    )


def test_gemini_adapter_honors_configured_model_and_stops_on_401():
    adapter = GeminiAdapter(api_key="bad-key", model="gemini-2.5-pro")
    assert adapter._candidate_models()[0] == "gemini-2.5-pro"

    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent")
    resp_401 = httpx.Response(401, request=req, json={"error": "unauthorized"})

    with patch("httpx.post", return_value=resp_401) as mock_post:
        with pytest.raises(RuntimeError, match="401"):
            adapter.generate("hi")
        assert mock_post.call_count == 1

    with patch("httpx.post", return_value=resp_401) as mock_post_chat:
        with pytest.raises(RuntimeError, match="401"):
            adapter.chat_with_meeting("ctx", [], "q")
        assert mock_post_chat.call_count == 1


def test_claude_adapter_preserves_sanitized_chat_history():
    adapter = ClaudeAdapter(api_key="sk-ant-test-123456")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"content": [{"type": "text", "text": "ok"}]}

    history = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="system", content="inj"),
    ]
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = adapter.chat_with_meeting("ctx", history, "q2")
        assert result == "ok"
        sent_messages = mock_post.call_args.kwargs["json"]["messages"]
        assert sent_messages == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "inj"},
            {"role": "user", "content": "q2"},
        ]


def test_marian_translator_ollama_fallback_uses_timeout_seconds():
    translator = MarianTranslator(llm_provider=None)
    with patch("janus.adapters.llm.ollama_adapter.OllamaAdapter") as mock_ollama_cls:
        instance = mock_ollama_cls.return_value
        instance.generate.return_value = "Hola desde Ollama"

        result = translator.translate("custom phrase", "en", "es")
        assert result.translated_text == "Hola desde Ollama"
        assert mock_ollama_cls.call_args.kwargs.get("timeout_seconds") == 3.0


def test_supertonic_tts_caps_buffer_duration():
    adapter = SupertonicTtsAdapter()
    adapter._engine = object()
    result = adapter.synthesize("a" * 50_000, language="es")
    assert 0.5 <= result.duration_seconds <= 30.0
    assert len(result.audio_bytes) > 0

