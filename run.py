import os
import sys
import socket
import argparse
import uvicorn

# Ensure the project root directory is at the beginning of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import logging

# Configure root logger for Janus application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is already open and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def resolve_port(preferred_port: int, host: str = "127.0.0.1") -> int:
    """Check availability and fallback if conflict detected on Windows."""
    if not is_port_in_use(host, preferred_port):
        return preferred_port

    fallback_port = 8080 if preferred_port == 8000 else preferred_port + 1
    print("\n" + "=" * 65)
    print(f"[!] ALERTA: El puerto {preferred_port} ya está en uso por otro proceso.")
    print("    En Windows, esto causa que las peticiones vayan al otro servidor")
    print("    (por ejemplo OpenFolioLM) devolviendo '{\"detail\":\"Not Found\"}'.")
    print(f"[+] Cambiando automáticamente al puerto libre: {fallback_port}")
    print(f"[+] Abrí en tu navegador: http://{host}:{fallback_port}")
    print("=" * 65 + "\n")
    return fallback_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janus")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host address (default: 127.0.0.1)",
    )
    args, unknown = parser.parse_known_args()

    target_port = resolve_port(args.port, args.host)

    print(f"[*] Iniciando Janus en: {PROJECT_ROOT}")
    print(f"[*] Interfaz Web disponible en: http://{args.host}:{target_port}")

    uvicorn.run(
        "janus.api.app:app",
        host=args.host,
        port=target_port,
        reload=True,
        log_level="info",
        app_dir=PROJECT_ROOT,
    )

