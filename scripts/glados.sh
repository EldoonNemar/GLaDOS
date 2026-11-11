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

compose() { docker compose "$@"; }

# --profile must come BEFORE the subcommand (up/down/build/logs)
if [[ "$TARGET" == "spoke" ]]; then
    PROFILE=(--profile spoke)
fi

case "$ACTION" in
    start)
        echo "[glados] Starting $TARGET..."
        if [[ "$TARGET" == "spoke" ]]; then
            compose ${PROFILE[@]} up -d glados_spoke
        else
            compose up -d glados_llama glados_engine glados_hub
            echo "[glados] Backend up. Hub: http://localhost:5050/health"
        fi
        ;;
    stop)
        echo "[glados] Stopping $TARGET..."
        if [[ "$TARGET" == "spoke" ]]; then
            compose down glados_spoke
        else
            compose down glados_llama glados_engine glados_hub
        fi
        ;;
    restart)
        echo "[glados] Restarting $TARGET..."
        if [[ "$TARGET" == "spoke" ]]; then
            compose down glados_spoke && compose ${PROFILE[@]} up -d glados_spoke
        else
            compose down glados_llama glados_engine glados_hub && compose up -d glados_llama glados_engine glados_hub
            echo "[glados] Backend up. Hub: http://localhost:5050/health"
        fi
        ;;
    build)
        echo "[glados] Building $TARGET..."
        if [[ "$TARGET" == "spoke" ]]; then
            compose ${PROFILE[@]} build glados_spoke
        else
            compose build glados_engine glados_hub
        fi
        ;;
    logs)
        echo "[glados] Showing logs for $TARGET..."
        if [[ "$TARGET" == "spoke" ]]; then
            compose logs -f glados_spoke
        else
            compose logs -f glados_llama glados_engine glados_hub
        fi
        ;;
esac
