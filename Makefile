# Load DB_DATA_DIR from .env so Make and Docker Compose never drift apart.
# Falls back to a local relative path if .env is missing or the key isn't found.
DB_DATA_DIR := $(shell grep -s '^DB_DATA_DIR=' .env | cut -d '=' -f2-)
ifndef DB_DATA_DIR
    DB_DATA_DIR := ./checkpoints/db
endif

.PHONY: help setup up down restart status logs pull clean

help:
	@echo "Available commands:"
	@echo "  make setup    - Create data directories and fix permissions"
	@echo "  make pull     - Pull all Docker images"
	@echo "  make up       - Start all services in the background"
	@echo "  make down     - Stop all services (data is preserved)"
	@echo "  make restart  - Restart all services"
	@echo "  make status   - Show running containers"
	@echo "  make logs     - Tail logs of all services"
	@echo "  make clean    - WARNING: Stop services and DELETE all local database data"

setup:
	@echo "📁 Ensuring base directory exists: $(DB_DATA_DIR)"
	@mkdir -p $(DB_DATA_DIR)
	@echo "📁 Creating database subdirectories..."
	@mkdir -p $(DB_DATA_DIR)/neo4j/data
	@mkdir -p $(DB_DATA_DIR)/neo4j/logs
	@mkdir -p $(DB_DATA_DIR)/qdrant
	@mkdir -p $(DB_DATA_DIR)/elasticsearch
	@mkdir -p $(DB_DATA_DIR)/postgres
	@mkdir -p $(DB_DATA_DIR)/redis
	@echo "🔒 Fixing Elasticsearch directory permissions..."
	@sudo chown -R 1000:1000 $(DB_DATA_DIR)/elasticsearch
	@echo "✅ Setup complete!"

pull:
	@echo "📥 Pulling latest Docker images..."
	docker compose pull

up: setup
	@echo "🚀 Starting services..."
	docker compose up -d
	@echo "✅ Services started! Check 'make status' to verify."

down:
	@echo "🛑 Stopping services..."
	docker compose down
	@echo "✅ Services stopped. Data preserved in $(DB_DATA_DIR)."

restart:
	@echo "🔄 Restarting services..."
	docker compose restart

status:
	@echo "📊 Current container status:"
	docker compose ps

logs:
	@echo "📜 Tailing logs (Press Ctrl+C to exit)..."
	docker compose logs -f

clean:
	@echo "⚠️  WARNING: This will DELETE ALL DATA in $(DB_DATA_DIR)!"
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || (echo "Aborted."; exit 1)
	@test -n "$(DB_DATA_DIR)" || (echo "DB_DATA_DIR is empty! Aborting."; exit 1)
	@test "$(DB_DATA_DIR)" != "/" || (echo "Refusing to delete root! Aborting."; exit 1)
	docker compose down -v
	rm -rf $(DB_DATA_DIR)/neo4j $(DB_DATA_DIR)/qdrant $(DB_DATA_DIR)/elasticsearch $(DB_DATA_DIR)/postgres $(DB_DATA_DIR)/redis
	@echo "✅ Cleanup complete."