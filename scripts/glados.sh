#!/usr/bin/env bash
# GLaDOS - Docker management script
# Usage: ./scripts/glados.sh {start|stop|restart|build|logs} [--spoke]
set -euo pipefail
cd "$(dirname "$0")/.."

SPEKE=false
ACTION=""
for arg in "$@"; do
    case "$arg" in
        start|stop|restart|build|logs) ACTION="$arg" ;;
        --spoke) SPOKE=true ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: $0 {start|stop|restart|build|logs} [--spoke]"
    echo ""
    echo "Commands:"
    echo "  start   - Start GLaDOS backend (LLM + ASR/TTS + hub)"
    echo "  stop    - Stop all GLaDOS containers"
    echo "  restart - Stop then start"
    echo "  build   - Rebuild all Docker images"
    echo "  logs    - Follow logs from all containers"
    echo ""
    echo "Options:"
    echo "  --spoke  - Also build/start the Python spoke"
    echo ""
    echo "Examples:"
    echo "  ./scripts/glados.sh start            # Backend only"
    echo "  ./scripts/glados.sh start --spoke    # Backend + Python spoke"
    echo "  ./scripts/glados.sh stop             # Stop everything"
    echo "  ./scripts/glados.sh logs --spoke     # Follow spoke logs too"
    exit 0
fi

COMPOSE_CMD="docker compose"
COMPOSE_ARGS=()

if [[ "$SPEKE" == "true" ]]; then
    COMPOSE_ARGS+=(--profile spoke)
    echo "[glados] Building/starting with spoke..."
else
    echo "[glados] Building/starting backend only..."
fi

case "$ACTION" in
    start)
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} up -d glados_llama glados_engine glados_hub
        [[ "$SPEKE" == "true" ]] && $COMPOSE_CMD ${COMPOSE_ARGS[@]} up -d glados_spoke
        echo "[glados] Backend up. Hub: http://localhost:5050/health"
        ;;
    stop)
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} down
        echo "[glados] All containers stopped."
        ;;
    restart)
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} down
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} up -d glados_llama glados_engine glados_hub
        [[ "$SPEKE" == "true" ]] && $COMPOSE_CMD ${COMPOSE_ARGS[@]} up -d glados_spoke
        echo "[glados] Restarted. Hub: http://localhost:5050/health"
        ;;
    build)
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} build glados_engine glados_hub
        [[ "$SPEKE" == "true" ]] && $COMPOSE_CMD ${COMPOSE_ARGS[@]} build glados_spoke
        echo "[glados] Build complete."
        ;;
    logs)
        $COMPOSE_CMD ${COMPOSE_ARGS[@]} logs -f
        ;;
esac
