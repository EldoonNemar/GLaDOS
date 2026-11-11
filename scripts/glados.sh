#!/usr/bin/env bash
# GLaDOS - Docker management script
# Usage: ./scripts/glados.sh {start|stop|restart|build|logs} [backend|spoke]
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION=""
TARGET=""  # backend, spoke, or empty for all
for arg in "$@"; do
    case "$arg" in
        start|stop|restart|build|logs) ACTION="$arg" ;;
        backend|spoke) TARGET="$arg" ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: $0 {start|stop|restart|build|logs} [backend|spoke]"
    echo ""
    echo "Commands:"
    echo "  start   - Start GLaDOS backend or spoke"
    echo "  stop    - Stop all GLaDOS containers"
    echo "  restart - Stop then start"
    echo "  build   - Rebuild Docker images"
    echo "  logs    - Follow logs"
    echo ""
    echo "Targets (default: both):"
    echo "  backend - Only LLM + ASR/TTS + hub (the server)"
    echo "  spoke   - Only the Python spoke (connects to a remote hub)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/glados.sh start              # Start backend"
    echo "  ./scripts/glados.sh start spoke        # Start spoke only"
    echo "  ./scripts/glados.sh start backend      # Start backend only"
    echo "  ./scripts/glados.sh stop               # Stop everything"
    echo ""
    echo "Note: The spoke connects to a remote hub. Set GLADOS_HUB_URL"
    echo "      in the spoke container to point to your backend server."
    exit 0
fi

case "$TARGET" in
    spoke)
        echo "[glados] Targeting spoke only..."
        case "$ACTION" in
            start)  docker compose up -d --profile spoke glados_spoke ;;
            stop)   docker compose down glados_spoke ;;
            restart) docker compose down glados_spoke && docker compose up -d --profile spoke glados_spoke ;;
            build)  docker compose build glados_spoke ;;
            logs)   docker compose logs -f glados_spoke ;;
        esac
        ;;
    backend)
        echo "[glados] Targeting backend only..."
        case "$ACTION" in
            start)  docker compose up -d glados_llama glados_engine glados_hub ;;
            stop)   docker compose down glados_llama glados_engine glados_hub ;;
            restart) docker compose down glados_llama glados_engine glados_hub && docker compose up -d glados_llama glados_engine glados_hub ;;
            build)  docker compose build glados_engine glados_hub ;;
            logs)   docker compose logs -f glados_llama glados_engine glados_hub ;;
        esac
        echo "[glados] Backend up. Hub: http://localhost:5050/health"
        ;;
    "")
        echo "[glados] Starting backend..."
        case "$ACTION" in
            start)  docker compose up -d glados_llama glados_engine glados_hub ;;
            stop)   docker compose down glados_llama glados_engine glados_hub ;;
            restart) docker compose down glados_llama glados_engine glados_hub && docker compose up -d glados_llama glados_engine glados_hub ;;
            build)  docker compose build glados_engine glados_hub ;;
            logs)   docker compose logs -f glados_llama glados_engine glados_hub ;;
        esac
        echo "[glados] Backend up. Hub: http://localhost:5050/health"
        ;;
esac
