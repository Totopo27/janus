from typing import List, Optional
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field
from janus.domain.models import SpeakerProfile, Meeting
from janus.services.session_service import SessionService
from janus.services.meeting_notes_service import MeetingNotesService
from janus.ports.storage_port import IMeetingRepository


class SpeakerProfileSchema(BaseModel):
    speaker_id: str
    name: str
    native_language: str
    preferred_voice_style: str = "default"


class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    speaker_a: SpeakerProfileSchema
    speaker_b: SpeakerProfileSchema


class SessionResponse(BaseModel):
    session_id: str
    speaker_a: SpeakerProfileSchema
    speaker_b: SpeakerProfileSchema
    turn_count: int
    created_at: float


class CreateMeetingRequest(BaseModel):
    meeting_id: str = Field(..., description="Unique meeting identifier")
    title: str = Field(..., description="Meeting topic or title")
    speaker_a: SpeakerProfileSchema
    speaker_b: SpeakerProfileSchema


class ActionItemSchema(BaseModel):
    assignee: str
    task: str
    completed: bool = False
    due_hint: Optional[str] = None


class MeetingSummarySchema(BaseModel):
    executive_summary: str
    key_points: List[str]
    action_items: List[ActionItemSchema]
    generated_at: float


class MeetingResponse(BaseModel):
    meeting_id: str
    title: str
    speaker_a: SpeakerProfileSchema
    speaker_b: SpeakerProfileSchema
    status: str
    turn_count: int
    created_at: float
    ended_at: Optional[float] = None
    summary: Optional[MeetingSummarySchema] = None


def create_api_router(
    session_service: SessionService,
    meeting_repo: Optional[IMeetingRepository] = None,
    notes_service: Optional[MeetingNotesService] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Sessions & Meetings"])

    # -------------------------------------------------------------
    # Session Routes (In-Memory Live Sessions)
    # -------------------------------------------------------------

    @router.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session(request: CreateSessionRequest):
        spk_a = SpeakerProfile(
            speaker_id=request.speaker_a.speaker_id,
            name=request.speaker_a.name,
            native_language=request.speaker_a.native_language,
            preferred_voice_style=request.speaker_a.preferred_voice_style,
        )
        spk_b = SpeakerProfile(
            speaker_id=request.speaker_b.speaker_id,
            name=request.speaker_b.name,
            native_language=request.speaker_b.native_language,
            preferred_voice_style=request.speaker_b.preferred_voice_style,
        )
        session = session_service.create_session(
            session_id=request.session_id,
            speaker_a=spk_a,
            speaker_b=spk_b,
        )

        # Also auto-register in meeting repo if active
        if meeting_repo:
            existing = meeting_repo.get_meeting(request.session_id)
            if not existing:
                meeting_repo.save_meeting(
                    Meeting(
                        meeting_id=request.session_id,
                        title=f"Reunión {spk_a.name} & {spk_b.name}",
                        speaker_a=spk_a,
                        speaker_b=spk_b,
                    )
                )

        return SessionResponse(
            session_id=session.session_id,
            speaker_a=request.speaker_a,
            speaker_b=request.speaker_b,
            turn_count=len(session.turns),
            created_at=session.created_at,
        )

    @router.get("/sessions", response_model=List[SessionResponse])
    def list_sessions():
        sessions = session_service.list_sessions()
        return [
            SessionResponse(
                session_id=s.session_id,
                speaker_a=SpeakerProfileSchema(
                    speaker_id=s.speaker_a.speaker_id,
                    name=s.speaker_a.name,
                    native_language=s.speaker_a.native_language,
                    preferred_voice_style=s.speaker_a.preferred_voice_style,
                ),
                speaker_b=SpeakerProfileSchema(
                    speaker_id=s.speaker_b.speaker_id,
                    name=s.speaker_b.name,
                    native_language=s.speaker_b.native_language,
                    preferred_voice_style=s.speaker_b.preferred_voice_style,
                ),
                turn_count=len(s.turns),
                created_at=s.created_at,
            )
            for s in sessions
        ]

    @router.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str):
        s = session_service.get_session(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(
            session_id=s.session_id,
            speaker_a=SpeakerProfileSchema(
                speaker_id=s.speaker_a.speaker_id,
                name=s.speaker_a.name,
                native_language=s.speaker_a.native_language,
                preferred_voice_style=s.speaker_a.preferred_voice_style,
            ),
            speaker_b=SpeakerProfileSchema(
                speaker_id=s.speaker_b.speaker_id,
                name=s.speaker_b.name,
                native_language=s.speaker_b.native_language,
                preferred_voice_style=s.speaker_b.preferred_voice_style,
            ),
            turn_count=len(s.turns),
            created_at=s.created_at,
        )

    @router.get("/sessions/{session_id}/transcript")
    def get_transcript(session_id: str):
        s = session_service.get_session(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_service.export_transcript(session_id)

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        closed = session_service.close_session(session_id)
        if not closed:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "closed": True}

    # -------------------------------------------------------------
    # Meeting & Zoom-Style Notes Routes (Persistent SQLite Storage)
    # -------------------------------------------------------------

    @router.post(
        "/meetings",
        response_model=MeetingResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_meeting(request: CreateMeetingRequest):
        if not meeting_repo:
            raise HTTPException(status_code=503, detail="Storage repository not configured")

        spk_a = SpeakerProfile(
            speaker_id=request.speaker_a.speaker_id,
            name=request.speaker_a.name,
            native_language=request.speaker_a.native_language,
            preferred_voice_style=request.speaker_a.preferred_voice_style,
        )
        spk_b = SpeakerProfile(
            speaker_id=request.speaker_b.speaker_id,
            name=request.speaker_b.name,
            native_language=request.speaker_b.native_language,
            preferred_voice_style=request.speaker_b.preferred_voice_style,
        )
        meeting = Meeting(
            meeting_id=request.meeting_id,
            title=request.title,
            speaker_a=spk_a,
            speaker_b=spk_b,
        )
        meeting_repo.save_meeting(meeting)

        # Also register in session service so audio can be streamed immediately
        session_service.create_session(
            session_id=request.meeting_id,
            speaker_a=spk_a,
            speaker_b=spk_b,
        )

        return _build_meeting_response(meeting)

    @router.get("/meetings", response_model=List[MeetingResponse])
    def list_meetings():
        if not meeting_repo:
            raise HTTPException(status_code=503, detail="Storage repository not configured")
        meetings = meeting_repo.list_meetings()
        return [_build_meeting_response(m) for m in meetings]

    @router.get("/meetings/{meeting_id}", response_model=MeetingResponse)
    def get_meeting(meeting_id: str):
        if not meeting_repo:
            raise HTTPException(status_code=503, detail="Storage repository not configured")
        m = meeting_repo.get_meeting(meeting_id)
        if not m:
            raise HTTPException(status_code=404, detail="Meeting not found")
        return _build_meeting_response(m)

    @router.post("/meetings/{meeting_id}/finalize", response_model=MeetingResponse)
    def finalize_meeting(meeting_id: str):
        """
        Finalizes a meeting and automatically generates Zoom-style executive summary,
        key points, and action items.
        """
        if not notes_service or not meeting_repo:
            raise HTTPException(status_code=503, detail="Meeting notes service not configured")

        summary = notes_service.generate_notes_and_finalize(meeting_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Meeting not found")

        updated = meeting_repo.get_meeting(meeting_id)
        return _build_meeting_response(updated)

    @router.get("/meetings/{meeting_id}/notes")
    def get_meeting_notes(meeting_id: str, format: str = "json"):
        """
        Retrieves the meeting notes in JSON or raw GitHub Flavored Markdown format.
        """
        if not notes_service or not meeting_repo:
            raise HTTPException(status_code=503, detail="Meeting notes service not configured")

        m = meeting_repo.get_meeting(meeting_id)
        if not m:
            raise HTTPException(status_code=404, detail="Meeting not found")

        if format.lower() == "markdown" or format.lower() == "md":
            markdown = notes_service.export_as_markdown(meeting_id)
            return Response(
                content=markdown,
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{meeting_id}_notes.md"'}
            )

        if not m.summary:
            # Auto-generate if not yet generated
            notes_service.generate_notes_and_finalize(meeting_id)
            m = meeting_repo.get_meeting(meeting_id)

        return _build_meeting_response(m).summary

    return router


def _build_meeting_response(m: Meeting) -> MeetingResponse:
    summary_schema = None
    if m.summary:
        summary_schema = MeetingSummarySchema(
            executive_summary=m.summary.executive_summary,
            key_points=m.summary.key_points,
            action_items=[
                ActionItemSchema(
                    assignee=i.assignee,
                    task=i.task,
                    completed=i.completed,
                    due_hint=i.due_hint,
                )
                for i in m.summary.action_items
            ],
            generated_at=m.summary.generated_at,
        )

    return MeetingResponse(
        meeting_id=m.meeting_id,
        title=m.title,
        speaker_a=SpeakerProfileSchema(
            speaker_id=m.speaker_a.speaker_id,
            name=m.speaker_a.name,
            native_language=m.speaker_a.native_language,
            preferred_voice_style=m.speaker_a.preferred_voice_style,
        ),
        speaker_b=SpeakerProfileSchema(
            speaker_id=m.speaker_b.speaker_id,
            name=m.speaker_b.name,
            native_language=m.speaker_b.native_language,
            preferred_voice_style=m.speaker_b.preferred_voice_style,
        ),
        status=m.status,
        turn_count=len(m.turns),
        created_at=m.created_at,
        ended_at=m.ended_at,
        summary=summary_schema,
    )
