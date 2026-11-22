#!/usr/bin/env bash
# GLaDOS - Docker management script
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION=""
TARGET=""
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
    echo "Targets (default: backend):"
    echo "  backend - Only LLM + ASR/TTS + hub (the server)"
    echo "  spoke   - Only the Python spoke (connects to a remote hub)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/glados.sh start              # Start backend"
    echo "  ./scripts/glados.sh start spoke        # Start spoke only"
    echo "  ./scripts/glados.sh start backend      # Start backend only"
    echo "  ./scripts/glados.sh stop               # Stop everything"
    exit 0
fi

case "$ACTION" in
    start)
        if [[ "$TARGET" == "spoke" ]]; then
            echo "[glados] Starting spoke only..."
            if [[ -f "docker/docker-compose-spoke.yml" ]]; then
                docker compose -f docker/docker-compose-spoke.yml --profile spoke up -d
            else
                docker compose up -d glados_spoke
            fi
        else
            echo "[glados] Starting backend..."
            docker compose up -d glados_llama glados_engine glados_hub
            echo "[glados] Backend up. Hub: http://localhost:5050/health"
        fi
        ;;
    stop)
        if [[ "$TARGET" == "spoke" ]]; then
            echo "[glados] Stopping spoke..."
            if [[ -f "docker/docker-compose-spoke.yml" ]]; then
                docker compose -f docker/docker-compose-spoke.yml down glados_spoke
            else
                docker compose down glados_spoke
            fi
        else
            echo "[glados] Stopping backend..."
            docker compose down glados_llama glados_engine glados_hub
        fi
        ;;
    restart)
        if [[ "$TARGET" == "spoke" ]]; then
            echo "[glados] Restarting spoke..."
            if [[ -f "docker/docker-compose-spoke.yml" ]]; then
                docker compose -f docker/docker-compose-spoke.yml down glados_spoke && docker compose -f docker/docker-compose-spoke.yml up -d glados_spoke
            else
                docker compose down glados_spoke && docker compose up -d glados_spoke
            fi
        else
            echo "[glados] Restarting backend..."
            docker compose down glados_llama glados_engine glados_hub && docker compose up -d glados_llama glados_engine glados_hub
            echo "[glados] Backend up. Hub: http://localhost:5050/health"
        fi
        ;;
    build)
        if [[ "$TARGET" == "spoke" ]]; then
            echo "[glados] Building spoke..."
            # Use spoke-only compose file if available (Orange Pi / arm64)
            if [[ -f "docker/docker-compose-spoke.yml" ]]; then
                cd docker && docker compose -f docker-compose-spoke.yml build glados_spoke && cd ..
            else
                docker compose build glados_spoke
            fi
        else
            echo "[glados] Building backend..."
            docker compose build glados_engine glados_hub
        fi
        ;;
    logs)
        if [[ "$TARGET" == "spoke" ]]; then
            docker compose logs -f glados_spoke
        else
            docker compose logs -f glados_llama glados_engine glados_hub
        fi
        ;;
esac
