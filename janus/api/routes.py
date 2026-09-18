from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from janus.domain.models import SpeakerProfile
from janus.services.session_service import SessionService


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


def create_api_router(session_service: SessionService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Sessions"])

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

    return router
