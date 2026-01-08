"""FastAPI application for SkimLit API."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import tensorflow as tf
import numpy as np
import time
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import TextPreprocessor
from src.utils.config import ConfigManager
from api.schemas import (
    AbstractRequest,
    AbstractResponse,
    SentencePrediction,
    BatchAbstractRequest,
    BatchAbstractResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)

app = FastAPI(
    title="SkimLit API",
    description="AI-powered RCT abstract structuring API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_COUNT = Counter("skimlit_requests_total", "Total requests")
REQUEST_LATENCY = Histogram("skimlit_request_duration_seconds", "Request latency")
SENTENCE_COUNT = Counter("skimlit_sentences_processed_total", "Total sentences processed")

class AppState:
    """Application state container."""
    model = None
    preprocessor = None
    config = None
    start_time = time.time()
    stats = {
        "total_requests": 0,
        "total_sentences": 0,
        "total_processing_time": 0
    }

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup."""
    logger.info("Starting SkimLit API...")

    try:
        config_manager = ConfigManager("configs/model_config.yaml")
        state.config = config_manager.config
        logger.info("Configuration loaded")

        state.preprocessor = TextPreprocessor(state.config.get("data", {}).get("preprocessing", {}))
        state.preprocessor.load_spacy_model()
        logger.info("Preprocessor initialized")

        model_path = "models/skimlit_tribrid_model"
        if Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            state.model = tf.keras.models.load_model(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}, API will run without model")
            state.model = None

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        state.model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down SkimLit API...")


def get_model():
    """Dependency to get model."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.model


def get_preprocessor():
    """Dependency to get preprocessor."""
    if state.preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    return state.preprocessor


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SkimLit API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        version="2.0.0"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics."""
    uptime = time.time() - state.start_time
    avg_time = (
        state.stats["total_processing_time"] / state.stats["total_requests"]
        if state.stats["total_requests"] > 0
        else 0
    )

    return StatsResponse(
        total_requests=state.stats["total_requests"],
        total_sentences=state.stats["total_sentences"],
        avg_processing_time=avg_time,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=AbstractResponse)
async def predict_abstract(
    request: AbstractRequest,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor)
):
    """
    Analyze an RCT abstract and classify each sentence.
    """
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        processed_data = preprocessor.prepare_inference_data(request.text)

        sentences = processed_data["sentences"]
        char_sequences = processed_data["char_sequences"]
        line_numbers_one_hot = processed_data["line_numbers_one_hot"]
        total_lines_one_hot = processed_data["total_lines_one_hot"]

        predictions = model.predict(
            {
                "line_number_input": line_numbers_one_hot,
                "total_lines_input": total_lines_one_hot,
                "token_input": np.array(sentences),
                "char_input": np.array(char_sequences)
            },
            verbose=0
        )

        pred_labels = preprocessor.decode_predictions(predictions)
        max_probs = np.max(predictions, axis=1)

        sentence_predictions = []
        label_names = state.config.get("data", {}).get("labels", ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"])
        for idx, (sentence, label, prob) in enumerate(zip(sentences, pred_labels, max_probs)):
            pred = SentencePrediction(
                text=sentence,
                label=label,
                confidence=float(prob),
                line_number=idx
            )

            if request.return_probabilities:
                pred.probabilities = {
                    name: float(predictions[idx][i]) if i < len(predictions[idx]) else 0.0
                    for i, name in enumerate(label_names)
                }

            sentence_predictions.append(pred)

        processing_time = time.time() - start_time

        state.stats["total_requests"] += 1
        state.stats["total_sentences"] += len(sentences)
        state.stats["total_processing_time"] += processing_time

        SENTENCE_COUNT.inc(len(sentences))
        REQUEST_LATENCY.observe(processing_time)

        return AbstractResponse(
            sentences=sentence_predictions,
            total_sentences=len(sentences),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchAbstractResponse)
async def batch_predict(
    request: BatchAbstractRequest,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor)
):
    """
    Batch process multiple abstracts.
    """
    start_time = time.time()

    results = []
    for abstract_request in request.abstracts:
        try:
            result = await predict_abstract(abstract_request, model, preprocessor)
            results.append(result)
        except Exception as e:
            logger.warning(f"Batch prediction skipped an abstract: {e}")
            continue

    total_time = time.time() - start_time

    return BatchAbstractResponse(
        results=results,
        total_abstracts=len(results),
        total_processing_time=total_time
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
