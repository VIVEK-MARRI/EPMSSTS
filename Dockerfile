# Multi-stage Dockerfile for EPMSSTS
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY epmssts/ epmssts/
COPY frontend/ frontend/

# Create outputs directory
RUN mkdir -p outputs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Run API server
CMD ["uvicorn", "epmssts.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
