# ============================
# 1️⃣ BUILD STAGE
# ============================
FROM python:3.10.14-slim-bullseye AS builder

# Set working directory
WORKDIR /application

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install system-level dependencies (including PortAudio for sounddevice)
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libportaudio2 \
        libasound-dev \
        ffmpeg \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Clean up to keep image small
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*


# ============================
# 2️⃣ RUNTIME STAGE
# ============================
FROM python:3.10.14-slim-bullseye

# Set working directory
WORKDIR /application

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install only runtime audio dependencies (lighter than dev)
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy trained model (make sure it exists locally)
COPY final_model/model.pt /final_model/model.pt

# Copy application source code
COPY . .

# Optional: pre-warm large imports to reduce cold start on EC2
USER root
RUN python -c "import torch; import pandas; import numpy; import sklearn; import flask" \
    && chown -R appuser:appuser /application
USER appuser

# Expose Flask default port
EXPOSE 5000

# Start your app
CMD ["python", "application.py"]

