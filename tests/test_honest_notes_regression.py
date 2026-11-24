"""Regression probes for the honest meeting notes change (S1-S5)."""
import hashlib
import json
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.stt.mock_stt_adapter import MockSpeechRecognizer
from janus.adapters.translation.mock_translator import MockTranslator
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.adapters.tts.mock_tts_adapter import MockSpeechSynthesizer
from janus.api.app import create_app
from janus.domain.models import (
    ConversationTurn,
    Meeting,
    MeetingSummary,
    SpeakerProfile,
    TranscriptionResult,
    TranslationResult,
)
from janus.services.meeting_notes_service import MeetingNotesService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.session_service import SessionService

OLD_DDL = """
CREATE TABLE meetings (meeting_id TEXT PRIMARY KEY, title TEXT NOT NULL, topic_key TEXT,
 speaker_a_id TEXT NOT NULL, speaker_a_name TEXT NOT NULL, speaker_a_lang TEXT NOT NULL,
 speaker_a_voice TEXT NOT NULL DEFAULT 'default', speaker_b_id TEXT NOT NULL,
 speaker_b_name TEXT NOT NULL, speaker_b_lang TEXT NOT NULL,
 speaker_b_voice TEXT NOT NULL DEFAULT 'default', status TEXT NOT NULL DEFAULT 'active',
 created_at REAL NOT NULL, ended_at REAL);
CREATE TABLE turns (turn_id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL, speaker_id TEXT NOT NULL,
 original_text TEXT NOT NULL, source_lang TEXT NOT NULL, translated_text TEXT NOT NULL,
 target_lang TEXT NOT NULL, latency_ms REAL NOT NULL DEFAULT 0.0,
 start_time REAL NOT NULL DEFAULT 0.0, end_time REAL NOT NULL DEFAULT 0.0,
 confidence REAL NOT NULL DEFAULT 1.0, created_at REAL NOT NULL);
CREATE TABLE summaries (meeting_id TEXT PRIMARY KEY, executive_summary TEXT NOT NULL,
 key_points_json TEXT NOT NULL DEFAULT '[]', generated_at REAL NOT NULL);
CREATE TABLE action_items (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL,
 assignee TEXT NOT NULL, task TEXT NOT NULL, due_hint TEXT, completed INTEGER NOT NULL DEFAULT 0);
"""

HALLUCINATION_MARKERS = (
    "establecieron compromisos",
    "Revisar acuerdos",
    "coordinaron puntos técnicos",
    "Conversación bilingüe",
    "Participante",
    "Seguimiento",
)

EXCERPTS = [
    "Alice: The quarterly figures look stable this month.",
    "Bob: Me encargo de enviar el resumen el viernes.",
]


def seeded(db, mid="m"):
    repo = SqliteMeetingRepository(db)
    repo.save_meeting(
        Meeting(
            mid,
            "M",
            SpeakerProfile("a", "Alice", "en"),
            SpeakerProfile("b", "Bob", "es"),
        )
    )
    repo.save_turn(
        mid,
        ConversationTurn(
            mid + "-1",
            mid,
            "a",
            TranscriptionResult(EXCERPTS[0][7:], "en"),
            TranslationResult(
                EXCERPTS[0][7:],
                "en",
                "Las cifras trimestrales se ven estables este mes.",
                "es",
            ),
        ),
    )
    repo.save_turn(
        mid,
        ConversationTurn(
            mid + "-2",
            mid,
            "b",
            TranscriptionResult(EXCERPTS[1][5:], "es"),
            TranslationResult(
                EXCERPTS[1][5:],
                "es",
                "I will send the summary on Friday.",
                "en",
            ),
        ),
    )
    return repo


def rows(db, mid):
    conn = sqlite3.connect(db)
    payload = [
        [
            tuple(r)
            for r in conn.execute(
                f"SELECT * FROM {table} WHERE meeting_id=? ORDER BY 1", (mid,)
            )
        ]
        for table in ("meetings", "turns", "summaries", "action_items")
    ]
    conn.close()
    return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()


def test_s3_migration_is_additive_idempotent_and_marks_legacy(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(OLD_DDL)
    conn.execute(
        "INSERT INTO meetings VALUES ('old','Old','k','a','Alice','en','default',"
        "'b','Bob','es','default','completed',10.0,11.0)"
    )
    conn.execute(
        "INSERT INTO summaries VALUES ('old','Existing summary','[\"Existing point\"]',123.0)"
    )
    conn.commit()
    conn.close()

    repo = SqliteMeetingRepository(db)
    snapshot_conn = sqlite3.connect(db)
    snapshot = [
        tuple(r)
        for table in ("meetings", "summaries")
        for r in snapshot_conn.execute(f"SELECT * FROM {table} ORDER BY 1")
    ]
    snapshot_conn.close()

    meeting = repo.get_meeting("old")
    assert (
        meeting.summary.executive_summary,
        meeting.summary.key_points,
        meeting.summary.generated_at,
    ) == ("Existing summary", ["Existing point"], 123.0)
    assert (meeting.summary.summary_status, meeting.summary.degraded_reason) == (
        "legacy",
        None,
    )
    assert meeting.title == "Old" and meeting.topic_key == "k"

    columns_conn = sqlite3.connect(db)
    columns = [r[1] for r in columns_conn.execute("PRAGMA table_info(summaries)")]
    columns_conn.close()
    assert {"summary_status", "degraded_reason"} <= set(columns)

    SqliteMeetingRepository(db)
    after_conn = sqlite3.connect(db)
    after = [
        tuple(r)
        for table in ("meetings", "summaries")
        for r in after_conn.execute(f"SELECT * FROM {table} ORDER BY 1")
    ]
    after_conn.close()
    assert after == snapshot
    assert repo.list_meetings()[0].summary.summary_status == "legacy"


@pytest.mark.parametrize(
    "mode,reason,payload",
    [
        ("none", "no_llm_provider", None),
        ("raise", "llm_error", RuntimeError("secret connection details")),
        ("bad_json", "invalid_llm_response", '{"executive_summary": "Fabricated outcome only."}'),
    ],
)
def test_s1_degraded_notes_hold_no_invented_content(tmp_path, mode, reason, payload):
    repo = seeded(str(tmp_path / "deg.db"), "deg")
    if payload is None:
        provider = None
    elif isinstance(payload, Exception):
        provider = Mock(generate=Mock(side_effect=payload))
    else:
        provider = Mock(generate=Mock(return_value=payload))

    summary = MeetingNotesService(repo, provider).generate_notes_and_finalize("deg")

    assert (summary.summary_status, summary.degraded_reason) == ("unavailable", reason)
    assert summary.executive_summary == ""
    blob = json.dumps(
        [
            summary.executive_summary,
            summary.key_points,
            [(a.assignee, a.task) for a in summary.action_items],
        ],
        ensure_ascii=False,
    )
    assert not [marker for marker in HALLUCINATION_MARKERS if marker in blob]
    assert summary.key_points == EXCERPTS
    assert [(a.assignee, a.task) for a in summary.action_items] == [
        ("Bob", "Enviar el resumen el viernes.")
    ]


def test_s1_rejected_llm_call_leaves_other_rows_byte_identical(tmp_path):
    db = str(tmp_path / "inv.db")
    seeded(db, "keep").save_summary(
        "keep", MeetingSummary(executive_summary="Keep me", generated_at=1.0)
    )
    seeded(db, "target")
    before = rows(db, "keep")

    MeetingNotesService(
        SqliteMeetingRepository(db), Mock(generate=Mock(return_value="not json"))
    ).generate_notes_and_finalize("target")

    assert rows(db, "keep") == before


def test_s1_translated_commitment_is_not_an_action_item(tmp_path):
    repo = SqliteMeetingRepository(str(tmp_path / "tr.db"))
    repo.save_meeting(
        Meeting("tr", "TR", SpeakerProfile("a", "Alice", "en"), SpeakerProfile("b", "Bob", "es"))
    )
    repo.save_turn(
        "tr",
        ConversationTurn(
            "tr-1",
            "tr",
            "a",
            TranscriptionResult("The draft is ready for review.", "en"),
            TranslationResult(
                "The draft is ready for review.",
                "en",
                "Me encargo de enviar el borrador mañana",
                "es",
            ),
        ),
    )
    assert MeetingNotesService(repo).generate_notes_and_finalize("tr").action_items == []


@pytest.mark.parametrize(
    "assignee,task",
    [("Participante", "Seguimiento"), ("TBD", "TBD"), ("N/A", "Por definir")],
)
def test_s2_placeholder_action_item_is_rejected(tmp_path, assignee, task):
    repo = seeded(str(tmp_path / "ph.db"), "ph")
    provider = Mock(
        generate=Mock(
            return_value=json.dumps(
                {
                    "executive_summary": "Resumen.",
                    "key_points": [],
                    "action_items": [
                        {"assignee": assignee, "task": task, "due_hint": None}
                    ],
                }
            )
        )
    )

    summary = MeetingNotesService(repo, provider).generate_notes_and_finalize("ph")

    assert (summary.summary_status, summary.degraded_reason) == (
        "unavailable",
        "invalid_llm_response",
    )
    rendered = [(item.assignee, item.task) for item in summary.action_items]
    assert (assignee, task) not in rendered
    for rendered_assignee, rendered_task in rendered:
        assert rendered_assignee.strip().lower() != assignee.strip().lower()
        assert rendered_task.strip().lower() != task.strip().lower()


def test_s3_s4_api_metadata_and_manual_edit(tmp_path):
    repo = SqliteMeetingRepository(str(tmp_path / "api.db"))
    broadcaster = WebSocketBroadcaster()
    app = create_app(
        session_service=SessionService(),
        broadcaster=broadcaster,
        meeting_repo=repo,
        notes_service=MeetingNotesService(repo),
        orchestrator=PipelineOrchestrator(
            stt_engine=MockSpeechRecognizer(),
            translation_engine=MockTranslator(),
            tts_engine=MockSpeechSynthesizer(),
            broadcaster=broadcaster,
            storage_repo=repo,
        ),
    )
    client = TestClient(app)

    created = client.post(
        "/api/meetings",
        json={
            "meeting_id": "api",
            "title": "API",
            "speaker_a": {"speaker_id": "a", "name": "Alice", "native_language": "en"},
            "speaker_b": {"speaker_id": "b", "name": "Bob", "native_language": "es"},
        },
    )
    assert created.status_code == 201
    seeded(str(tmp_path / "api.db"), "api")

    finalized = client.post("/api/meetings/api/finalize").json()["summary"]
    assert (finalized["summary_status"], finalized["degraded_reason"]) == (
        "unavailable",
        "no_llm_provider",
    )
    assert client.get("/api/meetings/api").json()["summary"] == finalized
    assert client.get("/api/meetings").json()[0]["summary"] == finalized
    assert client.get("/api/meetings/api/notes").json() == finalized

    for edit in ({"key_points": ["Manual point"]}, {"executive_summary": "Manual summary"}):
        response = client.put("/api/meetings/api/summary", json=edit)
        assert (
            response.status_code,
            response.json()["summary"]["summary_status"],
            response.json()["summary"]["degraded_reason"],
        ) == (200, "unavailable", "no_llm_provider")


@pytest.mark.parametrize(
    "exporter",
    ["export_as_markdown", "export_as_txt"],
)
def test_s5_exports_warn_without_inventing(tmp_path, exporter):
    repo = seeded(str(tmp_path / "exp.db"), "exp")
    repo.save_summary(
        "exp",
        MeetingSummary(
            executive_summary="",
            key_points=[EXCERPTS[0]],
            summary_status="unavailable",
            degraded_reason="no_llm_provider",
        ),
    )
    out = getattr(MeetingNotesService(repo), exporter)("exp")
    assert "Síntesis automática no disponible" in out
    assert EXCERPTS[0] in out
    assert "Revisar acuerdos" not in out

    repo.save_summary("exp", MeetingSummary(executive_summary="Old", summary_status="legacy"))
    assert "Estado de generación desconocido" in getattr(MeetingNotesService(repo), exporter)("exp")
