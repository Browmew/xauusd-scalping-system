#!/bin/bash
set -e

# Wait for database to be ready
if [ -n "$DATABASE_URL" ] || [ -n "$POSTGRES_HOST" ]; then
    echo "Waiting for database..."
    while ! pg_isready -h ${POSTGRES_HOST:-timescaledb} -p ${POSTGRES_PORT:-5432} -U ${POSTGRES_USER:-trader}; do
        sleep 1
    done
    echo "Database is ready!"
fi

# Wait for Redis to be ready
if [ -n "$REDIS_URL" ] || [ -n "$REDIS_HOST" ]; then
    echo "Waiting for Redis..."
    while ! redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping; do
        sleep 1
    done
    echo "Redis is ready!"
fi

# Initialize directories
mkdir -p /app/data /app/logs /app/models /app/results

# Set permissions
chmod -R 755 /app/data /app/logs /app/models /app/results

# Run database migrations if needed
if [ "$1" = "migrate" ]; then
    echo "Running database migrations..."
    python -c "
from src.data.storage import DataStorage
import asyncio

async def init_db():
    storage = DataStorage()
    await storage.initialize()
    print('Database initialized successfully')

asyncio.run(init_db())
"
    shift
fi

# Execute the main command
exec "$@"