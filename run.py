import os
import sys
import uvicorn

# Ensure the project root directory is at the beginning of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print(f"🚀 Iniciando Janus S2ST & Zoom AI Companion en: {PROJECT_ROOT}")
    uvicorn.run(
        "janus.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=PROJECT_ROOT,
    )
