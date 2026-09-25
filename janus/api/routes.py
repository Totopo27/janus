from typing import List, Optional, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field
from janus.domain.models import SpeakerProfile, Meeting, ChatMessage, SearchResult
from janus.services.session_service import SessionService
from janus.services.meeting_notes_service import MeetingNotesService
from janus.services.meeting_chat_service import MeetingChatService
from janus.adapters.llm.factory import LLMProviderFactory
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
    topic_key: Optional[str] = Field(None, description="Hierarchical topic key inspired by Engram (e.g. proyectos/janus)")
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
    topic_key: Optional[str] = None
    speaker_a: SpeakerProfileSchema
    speaker_b: SpeakerProfileSchema
    status: str
    turn_count: int
    created_at: float
    ended_at: Optional[float] = None
    summary: Optional[MeetingSummarySchema] = None


class SearchResultSchema(BaseModel):
    meeting_id: str
    turn_id: str
    speaker_id: str
    original_text: str
    translated_text: str
    snippet: str
    topic_key: Optional[str] = None
    created_at: float


class MeetingChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = None


class MeetingChatResponse(BaseModel):
    meeting_id: str
    question: str
    answer: str
    provider: str


class LLMConfigRequest(BaseModel):
    provider: str  # "ollama" | "gemini" | "mock"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class LLMConfigResponse(BaseModel):
    provider: str
    model: Optional[str] = None
    healthy: bool


class LiveNotesResponse(BaseModel):
    meeting_id: str
    current_topic: str
    key_takeaways: List[str]
    action_items: List[ActionItemSchema]
    last_processed_turn_index: int
    updated_at: float


class CatchUpRequest(BaseModel):
    last_n_turns: int = 6


class CatchUpResponse(BaseModel):
    meeting_id: str
    summary: str


def create_api_router(
    session_service: SessionService,
    meeting_repo: Optional[IMeetingRepository] = None,
    notes_service: Optional[MeetingNotesService] = None,
    chat_service: Optional[MeetingChatService] = None,
    live_notetaker: Optional[Any] = None,
    llm_factory: Optional[LLMProviderFactory] = None,
    fusion_service: Optional[Any] = None,
    translation_engine: Optional[Any] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Sessions, Meetings & BYOM"])


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
                        title=f"Conversación {request.session_id}",
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
        responses = []
        for s in sessions:
            responses.append(
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
            )
        return responses

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
    def export_session_transcript(session_id: str):
        return session_service.export_transcript(session_id)


    @router.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        success = session_service.close_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "closed": True}

    # -------------------------------------------------------------
    # Meeting & Structured Notes Routes (Persistent SQLite Storage)
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
            topic_key=request.topic_key,
            speaker_a=spk_a,
            speaker_b=spk_b,
        )
        meeting_repo.save_meeting(meeting)

        # Also register in session service so audio can be streamed immediately
        # Only create if no active session exists — avoid losing accumulated in-memory turns
        if not session_service.get_session(request.meeting_id):
            session_service.create_session(
                session_id=request.meeting_id,
                speaker_a=spk_a,
                speaker_b=spk_b,
            )

        return _build_meeting_response(meeting)

    @router.get("/meetings/search", response_model=List[SearchResultSchema])
    def search_turns(q: str, topic: Optional[str] = None):
        """Full-text search across all meetings and dialogue turns using SQLite FTS5."""
        if not meeting_repo:
            raise HTTPException(status_code=503, detail="Storage repository not configured")
        results = meeting_repo.search_turns(query=q, topic_key=topic)
        return [
            SearchResultSchema(
                meeting_id=r.meeting_id,
                turn_id=r.turn_id,
                speaker_id=r.speaker_id,
                original_text=r.original_text,
                translated_text=r.translated_text,
                snippet=r.snippet,
                topic_key=r.topic_key,
                created_at=r.created_at,
            )
            for r in results
        ]

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
        Finalizes a meeting and automatically generates executive summary,
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
    def get_meeting_notes(meeting_id: str, notes_format: Literal["json", "markdown", "md"] = "json"):
        """
        Retrieves the meeting notes in JSON or raw GitHub Flavored Markdown format.
        """
        if not notes_service or not meeting_repo:
            raise HTTPException(status_code=503, detail="Meeting notes service not configured")

        m = meeting_repo.get_meeting(meeting_id)
        if not m:
            raise HTTPException(status_code=404, detail="Meeting not found")

        if notes_format in ("markdown", "md"):
            markdown = notes_service.export_as_markdown(meeting_id)
            return Response(
                content=markdown,
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{meeting_id}_notes.md"'}
            )

        if not m.summary:
            notes_service.generate_notes_and_finalize(meeting_id)
            m = meeting_repo.get_meeting(meeting_id)

        return _build_meeting_response(m).summary

    @router.post("/meetings/{meeting_id}/chat", response_model=MeetingChatResponse)
    def chat_with_meeting(meeting_id: str, request: MeetingChatRequest):
        """Conversational Q&A grounded on meeting transcript and summary via BYOM."""
        if not chat_service:
            raise HTTPException(status_code=503, detail="Meeting chat service not configured")

        history_msgs = []
        if request.history:
            for h in request.history:
                history_msgs.append(ChatMessage(role=h.get("role", "user"), content=h.get("content", "")))

        try:
            answer = chat_service.ask(
                meeting_id=meeting_id,
                question=request.question,
                history=history_msgs,
            )
            return MeetingChatResponse(
                meeting_id=meeting_id,
                question=request.question,
                answer=answer,
                provider=chat_service.llm_provider.provider_name,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    @router.get("/meetings/{meeting_id}/live-notes", response_model=LiveNotesResponse)
    def get_live_notes(meeting_id: str):
        """Returns real-time Janus structured notes (topic, takeaways, action items)."""
        if not live_notetaker:
            raise HTTPException(status_code=503, detail="Live notetaker service not configured")
        notes = live_notetaker.get_live_notes(meeting_id)
        return LiveNotesResponse(
            meeting_id=notes.meeting_id,
            current_topic=notes.current_topic,
            key_takeaways=notes.key_takeaways,
            action_items=[
                ActionItemSchema(
                    assignee=a.assignee,
                    task=a.task,
                    completed=a.completed,
                    due_hint=a.due_hint,
                )
                for a in notes.action_items
            ],
            last_processed_turn_index=notes.last_processed_turn_index,
            updated_at=notes.updated_at,
        )

    @router.post("/meetings/{meeting_id}/live-notes/refresh", response_model=LiveNotesResponse)
    def refresh_live_notes(meeting_id: str):
        """Forces an immediate synthesis update of live notes using LLM."""
        if not live_notetaker:
            raise HTTPException(status_code=503, detail="Live notetaker service not configured")
        notes = live_notetaker.update_notes(meeting_id)
        return LiveNotesResponse(
            meeting_id=notes.meeting_id,
            current_topic=notes.current_topic,
            key_takeaways=notes.key_takeaways,
            action_items=[
                ActionItemSchema(
                    assignee=a.assignee,
                    task=a.task,
                    completed=a.completed,
                    due_hint=a.due_hint,
                )
                for a in notes.action_items
            ],
            last_processed_turn_index=notes.last_processed_turn_index,
            updated_at=notes.updated_at,
        )

    @router.post("/meetings/{meeting_id}/catch-up", response_model=CatchUpResponse)
    def catch_up_meeting(meeting_id: str, request: CatchUpRequest = CatchUpRequest()):
        """Janus 'Catch Me Up': concise executive recap of the last few minutes."""
        if not live_notetaker:
            raise HTTPException(status_code=503, detail="Live notetaker service not configured")
        summary = live_notetaker.catch_up(meeting_id=meeting_id, last_n_turns=request.last_n_turns)
        return CatchUpResponse(
            meeting_id=meeting_id,
            summary=summary,
        )


    # -------------------------------------------------------------
    # BYOM System Configuration Routes
    # -------------------------------------------------------------

    @router.get("/system/llm-config", response_model=LLMConfigResponse)
    def get_llm_config():
        """Returns the active BYOM LLM provider and connectivity status."""
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not configured")
        provider = chat_service.llm_provider
        model = getattr(provider, "model", None)
        healthy = provider.health_check()
        return LLMConfigResponse(
            provider=provider.provider_name,
            model=model,
            healthy=healthy,
        )

    @router.post("/system/llm-config", response_model=LLMConfigResponse)
    def configure_llm(request: LLMConfigRequest):
        """Dynamically switches or reconfigures the active BYOM LLM provider (Ollama / Gemini)."""
        if not chat_service or not llm_factory:
            raise HTTPException(status_code=503, detail="Chat service or LLM factory not configured")

        config_dict = {}
        if request.api_key:
            config_dict["api_key"] = request.api_key
        if request.base_url:
            config_dict["base_url"] = request.base_url
        if request.model:
            config_dict["model"] = request.model

        new_provider = llm_factory.create(provider=request.provider, config=config_dict)
        chat_service.llm_provider = new_provider
        if live_notetaker and hasattr(live_notetaker, "llm_provider"):
            live_notetaker.llm_provider = new_provider
        if fusion_service and hasattr(fusion_service, "llm_provider"):
            fusion_service.llm_provider = new_provider
        if translation_engine and hasattr(translation_engine, "llm_provider"):
            translation_engine.llm_provider = new_provider
        model = getattr(new_provider, "model", None)
        healthy = new_provider.health_check()
        return LLMConfigResponse(
            provider=new_provider.provider_name,
            model=model,
            healthy=healthy,
        )

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
        topic_key=m.topic_key,
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
        turn_count=getattr(m, "_turn_count_cache", len(m.turns)),
        created_at=m.created_at,
        ended_at=m.ended_at,
        summary=summary_schema,
    )
