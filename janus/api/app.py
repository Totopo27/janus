import os
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
from fastapi.responses import FileResponse
from janus.api.routes import create_api_router
from janus.api.websockets import create_websocket_router
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.services.meeting_notes_service import MeetingNotesService
from janus.services.meeting_chat_service import MeetingChatService
from janus.services.live_notetaker_service import LiveNotetakerService
from janus.services.speaker_diarization_service import SpeakerDiarizationService
from janus.services.vad_segmenter import SileroVadSegmenter
from janus.services.conversational_fusion_service import ConversationalFusionService
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
    diarizer: Optional[SpeakerDiarizationService] = None,
    segmenter: Optional[SileroVadSegmenter] = None,
    db_path: str = "janus.db",
) -> FastAPI:
    """Factory creating and configuring the Janus FastAPI application."""
    app = FastAPI(
        title="Janus",
        description="Janus",
        version="0.2.0",
    )

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)

    # Allow local development, private IP ranges (LAN/Wi-Fi devices like tablets/phones)
    # matching standard RFC 1918 (192.168.x.x, 10.x.x.x, 172.16-31.x.x) and localhost.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$",
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
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            llm_provider = llm_factory.create("gemini", api_key=gemini_key, model="gemini-3.5-flash")
            logger.info("Janus LLM Intelligence active: Google Gemini Flash Cloud.")
        else:
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
        mt = MarianTranslator(llm_provider=llm_provider)
        tts = SupertonicTtsAdapter()
        if diarizer is None:
            # Check for Nemotron 3 Diarization ONNX model first for overlapped speech support
            from janus.services.nemotron_diarization_service import NemotronDiarizationService
            nemotron_service = NemotronDiarizationService()
            if nemotron_service._resolve_model_path():
                diarizer = nemotron_service
                logger.info("Janus acoustic diarization engine initialized: NVIDIA Nemotron 3 (Sortformer ONNX).")
            else:
                diar_threshold = float(os.environ.get("SPEAKER_SIMILARITY_THRESHOLD", "0.50"))
                diarizer = SpeakerDiarizationService(similarity_threshold=diar_threshold)
                logger.info("Janus acoustic diarization engine initialized: Sherpa-ONNX CAM++ / PyAnnote.")
        if segmenter is None:
            segmenter = SileroVadSegmenter()
        fusion_service = ConversationalFusionService(llm_provider=llm_provider, fallback_translator=mt)
        orchestrator = PipelineOrchestrator(
            stt_engine=stt,
            translation_engine=mt,
            tts_engine=tts,
            broadcaster=broadcaster,
            storage_repo=meeting_repo,
            live_notetaker=live_notetaker,
            diarizer=diarizer,
            segmenter=segmenter,
            fusion_service=fusion_service,
            enable_vad_slicing=True,
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
            fusion_service=fusion_service if 'fusion_service' in locals() else None,
            translation_engine=mt if 'mt' in locals() else None,
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
