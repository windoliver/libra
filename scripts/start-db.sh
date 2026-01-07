#!/usr/bin/env bash
###############################################################################
# Start LIBRA Database Services
#
# Usage:
#   ./scripts/start-db.sh          # Start and initialize
#   ./scripts/start-db.sh --stop   # Stop services
#   ./scripts/start-db.sh --reset  # Reset all data
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
case "${1:-}" in
  --stop)
    log_info "Stopping LIBRA services..."
    docker compose down
    log_info "Services stopped."
    exit 0
    ;;
  --reset)
    log_warn "This will DELETE all data. Are you sure? (y/N)"
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      log_info "Stopping services and removing volumes..."
      docker compose down -v
      log_info "Data reset complete."
    else
      log_info "Cancelled."
    fi
    exit 0
    ;;
  --help|-h)
    echo "Usage: $0 [--stop|--reset|--help]"
    echo ""
    echo "Options:"
    echo "  --stop   Stop all services"
    echo "  --reset  Stop and delete all data"
    echo "  --help   Show this help"
    exit 0
    ;;
esac

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  log_error "Docker is not running. Please start Docker first."
  exit 1
fi

# Start services
log_info "Starting LIBRA database services..."
docker compose up -d

# Wait for QuestDB to be healthy
log_info "Waiting for QuestDB to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s "http://localhost:9000/exec?query=SELECT%201" > /dev/null 2>&1; then
    log_info "QuestDB is ready!"
    break
  fi
  attempt=$((attempt + 1))
  echo -n "."
  sleep 1
done

if [ $attempt -eq $max_attempts ]; then
  log_error "QuestDB failed to start. Check logs with: docker compose logs questdb"
  exit 1
fi

echo ""

# Initialize tables
log_info "Initializing database tables..."
python3 -c "
import asyncio
from libra.data import AsyncQuestDBClient, QuestDBConfig

async def init():
    config = QuestDBConfig(username='libra', password='libra')
    async with AsyncQuestDBClient(config) as db:
        await db.create_tables()
        healthy = await db.health_check()
        if healthy:
            print('Tables created successfully!')
        else:
            print('Warning: Health check failed')

asyncio.run(init())
" 2>/dev/null || log_warn "Could not initialize tables (install with: pip install -e '.[database]')"

echo ""
log_info "======================================"
log_info "LIBRA Database Services Running"
log_info "======================================"
echo ""
echo "  QuestDB Web Console: http://localhost:9000"
echo "  QuestDB ILP Port:    localhost:9009"
echo "  QuestDB PG Port:     localhost:8812"
echo ""
echo "  Username: libra"
echo "  Password: libra"
echo ""
log_info "To stop: ./scripts/start-db.sh --stop"
log_info "To view logs: docker compose logs -f"
