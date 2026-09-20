import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from janus.api.routes import create_api_router
from janus.api.websockets import create_websocket_router
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.meeting_notes_service import MeetingNotesService
from janus.services.meeting_chat_service import MeetingChatService
from janus.services.live_notetaker_service import LiveNotetakerService
from janus.adapters.llm.factory import LLMProviderFactory
from janus.adapters.storage.sqlite_repository import SqliteMeetingRepository
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.adapters.stt.sherpa_stt_adapter import SherpaSttAdapter
from janus.adapters.translation.marian_translator import MarianTranslator
from janus.adapters.tts.supertonic_tts_adapter import SupertonicTtsAdapter
from janus.ports.storage_port import IMeetingRepository
from janus.ports.llm_port import ILLMProvider


def create_app(
    session_service: Optional[SessionService] = None,
    orchestrator: Optional[PipelineOrchestrator] = None,
    broadcaster: Optional[WebSocketBroadcaster] = None,
    meeting_repo: Optional[IMeetingRepository] = None,
    notes_service: Optional[MeetingNotesService] = None,
    chat_service: Optional[MeetingChatService] = None,
    live_notetaker: Optional[LiveNotetakerService] = None,
    llm_provider: Optional[ILLMProvider] = None,
    db_path: str = "janus.db",
) -> FastAPI:
    """Factory creating and configuring the Janus FastAPI application."""
    app = FastAPI(
        title="Janus",
        description="Janus",
        version="0.2.0",
    )

    # Local-only CORS: allow localhost on any port (dev) and file:// origins.
    # allow_origins=["*"] combined with allow_credentials=True is invalid per the
    # CORS spec and rejected by browsers — restrict to known local origins instead.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize default persistence & storage services
    if meeting_repo is None:
        meeting_repo = SqliteMeetingRepository(db_path=db_path)
    if notes_service is None:
        notes_service = MeetingNotesService(repository=meeting_repo)

    # Initialize default services if not injected
    if broadcaster is None:
        broadcaster = WebSocketBroadcaster()

    # Initialize BYOM & Chat service
    llm_factory = LLMProviderFactory()
    if llm_provider is None:
        llm_provider = llm_factory.create("ollama")
    if chat_service is None:
        chat_service = MeetingChatService(repository=meeting_repo, llm_provider=llm_provider)

    # Initialize Live Notetaker
    if live_notetaker is None:
        live_notetaker = LiveNotetakerService(
            repository=meeting_repo,
            llm_provider=llm_provider,
            broadcaster=broadcaster,
            batch_size=3,
        )

    if session_service is None:
        session_service = SessionService()
    if orchestrator is None:
        stt = SherpaSttAdapter()
        mt = MarianTranslator()
        tts = SupertonicTtsAdapter()
        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            broadcaster=broadcaster,
            storage_repo=meeting_repo,
            live_notetaker=live_notetaker,
        )

    # Health check
    @app.get("/health", tags=["System"])
    def health_check():
        return {"status": "ok", "app": "Janus", "version": "0.2.0"}

    # Include REST API & WebSockets
    app.include_router(
        create_api_router(
            session_service=session_service,
            meeting_repo=meeting_repo,
            notes_service=notes_service,
            chat_service=chat_service,
            live_notetaker=live_notetaker,
            llm_factory=llm_factory,
        )
    )

    app.include_router(
        create_websocket_router(
            session_service=session_service,
            orchestrator=orchestrator,
            broadcaster=broadcaster,
        )
    )


    # Serve static Web UI if directory exists
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def serve_ui():
            index_path = os.path.join(web_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Janus API is running. Web UI not found."}

    return app


# Default ASGI entrypoint
app = create_app()
