DB_DATA_DIR = /home/mehdi/Documents/projects/knowledge_graph_examples/QA_From_Your_Data/checkpoints/db

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
	@mkdir -p $(DB_DATA_DIR)/clickhouse
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
	@echo "⚠️  WARNING: This will DELETE ALL DATA!"
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || (echo "Aborted."; exit 1)
	docker compose down -v
	rm -rf $(DB_DATA_DIR)/*
	@echo "✅ Cleanup complete."
