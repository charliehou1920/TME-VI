#!/bin/bash
set -e

# Start FastAPI in background on port 8000
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to start..."
until curl -s http://localhost:8000/api/health > /dev/null; do
    sleep 1
done
echo "FastAPI ready."

# Start Streamlit on port 7860 (HF Spaces default)
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
