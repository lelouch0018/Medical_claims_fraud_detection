# Use a modern slim Python image
FROM python:3.12-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_BASE_DIR=/app \
    # set a blank default for the API key (override at runtime)
    OPENAI_API_KEY=""

WORKDIR /app

# Install small set of system deps often needed for packages and faiss
# keep list minimal to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgl1 \
      libc6 \
      libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install python deps
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create a non-root user and give ownership of the app directory
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose the app port
EXPOSE 8000

# Optional simple healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default command: run uvicorn (FastAPI)
# Note: override the OPENAI_API_KEY (or other envs) on docker run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
