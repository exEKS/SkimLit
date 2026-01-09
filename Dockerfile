# Multi-stage build for optimized SkimLit API image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Create directories for models and logs
RUN mkdir -p models logs

# Copy pre-trained model (if available locally)
# Uncomment if you have the model in your repo
# COPY models/skimlit_tribrid_model ./models/skimlit_tribrid_model

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TOKENIZERS_PARALLELISM=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 skimlit && \
    chown -R skimlit:skimlit /app

USER skimlit

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]