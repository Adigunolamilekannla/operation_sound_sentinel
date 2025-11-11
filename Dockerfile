# ---------------------------
# Build Stage
# ---------------------------
FROM python:3.10.14-slim-bullseye AS builder

# Set working directory
WORKDIR /application

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install build dependencies (including PortAudio for sounddevice)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential gcc portaudio19-dev \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Runtime Stage
# ---------------------------
FROM python:3.10.14-slim-bullseye

# Set working directory
WORKDIR /application

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install only runtime dependencies (PortAudio library needed at runtime)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


# Preload heavy libraries (optional)
USER root
RUN python -c "import pandas; import numpy; import sklearn; import flask; import torch" \
    && chown -R appuser:appuser /application

USER appuser

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "application.py"]
