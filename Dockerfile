# Multi-stage build for optimized SkimLit API image

# Stage 1: Builder
FROM python:3.9-slim as builder
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Stage 2: Runtime
FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages globally
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

RUN mkdir -p models logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

RUN useradd -m -u 1000 skimlit && \
    chown -R skimlit:skimlit /app

USER skimlit

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
