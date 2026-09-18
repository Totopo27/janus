import json
import logging
import sqlite3
import time
from typing import List, Optional
from janus.domain.models import (
    Meeting,
    SpeakerProfile,
    ConversationTurn,
    TranscriptionResult,
    TranslationResult,
    MeetingSummary,
    ActionItem,
)
from janus.ports.storage_port import IMeetingRepository

logger = logging.getLogger(__name__)


class SqliteMeetingRepository(IMeetingRepository):
    """
    Local-first SQLite repository implementing persistent storage
    for meetings, dialogue turns, speaker profiles, and Zoom-style meeting notes.
    """

    def __init__(self, db_path: str = "janus.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    speaker_a_id TEXT NOT NULL,
                    speaker_a_name TEXT NOT NULL,
                    speaker_a_lang TEXT NOT NULL,
                    speaker_a_voice TEXT NOT NULL DEFAULT 'default',
                    speaker_b_id TEXT NOT NULL,
                    speaker_b_name TEXT NOT NULL,
                    speaker_b_lang TEXT NOT NULL,
                    speaker_b_voice TEXT NOT NULL DEFAULT 'default',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    ended_at REAL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    start_time REAL NOT NULL DEFAULT 0.0,
                    end_time REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    meeting_id TEXT PRIMARY KEY,
                    executive_summary TEXT NOT NULL,
                    key_points_json TEXT NOT NULL DEFAULT '[]',
                    generated_at REAL NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS action_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL,
                    assignee TEXT NOT NULL,
                    task TEXT NOT NULL,
                    due_hint TEXT,
                    completed INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_meeting_id ON turns(meeting_id);
                CREATE INDEX IF NOT EXISTS idx_action_items_meeting_id ON action_items(meeting_id);
            """)

    def save_meeting(self, meeting: Meeting) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO meetings (
                    meeting_id, title,
                    speaker_a_id, speaker_a_name, speaker_a_lang, speaker_a_voice,
                    speaker_b_id, speaker_b_name, speaker_b_lang, speaker_b_voice,
                    status, created_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    ended_at=excluded.ended_at
            """, (
                meeting.meeting_id,
                meeting.title,
                meeting.speaker_a.speaker_id,
                meeting.speaker_a.name,
                meeting.speaker_a.native_language,
                meeting.speaker_a.preferred_voice_style,
                meeting.speaker_b.speaker_id,
                meeting.speaker_b.name,
                meeting.speaker_b.native_language,
                meeting.speaker_b.preferred_voice_style,
                meeting.status,
                meeting.created_at,
                meeting.ended_at,
            ))

    def save_turn(self, meeting_id: str, turn: ConversationTurn) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO turns (
                    turn_id, meeting_id, speaker_id,
                    original_text, source_lang,
                    translated_text, target_lang,
                    latency_ms, start_time, end_time, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn.turn_id,
                meeting_id,
                turn.speaker_id,
                turn.original_transcription.text,
                turn.original_transcription.language,
                turn.translation.translated_text,
                turn.translation.target_lang,
                turn.translation.latency_ms,
                turn.original_transcription.start_time,
                turn.original_transcription.end_time,
                turn.original_transcription.confidence,
                turn.created_at,
            ))

    def save_summary(self, meeting_id: str, summary: MeetingSummary) -> None:
        with self._get_connection() as conn:
            # 1. Update meeting status to completed
            conn.execute("""
                UPDATE meetings SET status = 'completed', ended_at = ? WHERE meeting_id = ?
            """, (time.time(), meeting_id))

            # 2. Insert or replace summary
            conn.execute("""
                INSERT OR REPLACE INTO summaries (
                    meeting_id, executive_summary, key_points_json, generated_at
                ) VALUES (?, ?, ?, ?)
            """, (
                meeting_id,
                summary.executive_summary,
                json.dumps(summary.key_points),
                summary.generated_at,
            ))

            # 3. Insert action items
            conn.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,))
            for item in summary.action_items:
                conn.execute("""
                    INSERT INTO action_items (meeting_id, assignee, task, due_hint, completed)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    meeting_id,
                    item.assignee,
                    item.task,
                    item.due_hint,
                    1 if item.completed else 0,
                ))

    def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        with self._get_connection() as conn:
            m_row = conn.execute("SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)).fetchone()
            if not m_row:
                return None

            speaker_a = SpeakerProfile(
                speaker_id=m_row["speaker_a_id"],
                name=m_row["speaker_a_name"],
                native_language=m_row["speaker_a_lang"],
                preferred_voice_style=m_row["speaker_a_voice"],
            )
            speaker_b = SpeakerProfile(
                speaker_id=m_row["speaker_b_id"],
                name=m_row["speaker_b_name"],
                native_language=m_row["speaker_b_lang"],
                preferred_voice_style=m_row["speaker_b_voice"],
            )

            # Retrieve turns ordered by created_at
            turn_rows = conn.execute(
                "SELECT * FROM turns WHERE meeting_id = ? ORDER BY created_at ASC",
                (meeting_id,)
            ).fetchall()

            turns = [
                ConversationTurn(
                    turn_id=t["turn_id"],
                    session_id=meeting_id,
                    speaker_id=t["speaker_id"],
                    original_transcription=TranscriptionResult(
                        text=t["original_text"],
                        language=t["source_lang"],
                        start_time=t["start_time"],
                        end_time=t["end_time"],
                        confidence=t["confidence"],
                    ),
                    translation=TranslationResult(
                        source_text=t["original_text"],
                        source_lang=t["source_lang"],
                        translated_text=t["translated_text"],
                        target_lang=t["target_lang"],
                        latency_ms=t["latency_ms"],
                    ),
                    created_at=t["created_at"],
                )
                for t in turn_rows
            ]

            # Retrieve summary if present
            s_row = conn.execute("SELECT * FROM summaries WHERE meeting_id = ?", (meeting_id,)).fetchone()
            summary = None
            if s_row:
                item_rows = conn.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,)).fetchall()
                action_items = [
                    ActionItem(
                        assignee=item["assignee"],
                        task=item["task"],
                        due_hint=item["due_hint"],
                        completed=bool(item["completed"]),
                    )
                    for item in item_rows
                ]
                summary = MeetingSummary(
                    executive_summary=s_row["executive_summary"],
                    key_points=json.loads(s_row["key_points_json"]),
                    action_items=action_items,
                    generated_at=s_row["generated_at"],
                )

            return Meeting(
                meeting_id=m_row["meeting_id"],
                title=m_row["title"],
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                turns=turns,
                summary=summary,
                status=m_row["status"],
                created_at=m_row["created_at"],
                ended_at=m_row["ended_at"],
            )

    def list_meetings(self) -> List[Meeting]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT meeting_id FROM meetings ORDER BY created_at DESC").fetchall()
            meetings = []
            for r in rows:
                m = self.get_meeting(r["meeting_id"])
                if m:
                    meetings.append(m)
            return meetings

    def delete_meeting(self, meeting_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM meetings WHERE meeting_id = ?", (meeting_id,))
            return cursor.rowcount > 0
