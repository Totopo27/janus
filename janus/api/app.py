import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from janus.api.routes import create_api_router
from janus.api.websockets import create_websocket_router
from janus.services.session_service import SessionService
from janus.services.pipeline_orchestrator import PipelineOrchestrator
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.adapters.stt.sherpa_stt_adapter import SherpaSttAdapter
from janus.adapters.translation.marian_translator import MarianTranslator
from janus.adapters.tts.supertonic_tts_adapter import SupertonicTtsAdapter


def create_app(
    session_service: SessionService = None,
    orchestrator: PipelineOrchestrator = None,
    broadcaster: WebSocketBroadcaster = None,
) -> FastAPI:
    """Factory creating and configuring the Janus FastAPI application."""
    app = FastAPI(
        title="Janus S2ST Platform",
        description="Local-First On-Device Speech-to-Speech Translation & Live Teleprompter",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize default services if not injected
    if broadcaster is None:
        broadcaster = WebSocketBroadcaster()
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
        )

    # Health check
    @app.get("/health", tags=["System"])
    def health_check():
        return {"status": "ok", "app": "Janus", "version": "0.1.0"}

    # Include REST API & WebSockets
    app.include_router(create_api_router(session_service))
    app.include_router(create_websocket_router(session_service, orchestrator, broadcaster))

    # Serve static Web UI if directory exists
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        def serve_ui():
            index_path = os.path.join(web_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Janus API is running. Web UI not found."}

    return app


# Default ASGI entrypoint
app = create_app()
