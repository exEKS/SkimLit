"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class AbstractRequest(BaseModel):
    """Request schema for abstract analysis."""
    
    text: str = Field(
        ...,
        description="Raw abstract text to analyze",
        min_length=10,
        max_length=10000
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to include prediction probabilities"
    )
    
    @validator("text")
    def text_not_empty(cls, v):
        """Validate text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This study examined the effects of a new drug on diabetes management. "
                       "A total of 200 patients were randomly assigned to treatment or control groups. "
                       "Blood glucose levels were measured weekly. Results showed a 25% reduction. "
                       "The new drug appears to be effective for diabetes management.",
                "return_probabilities": False
            }
        }


class SentencePrediction(BaseModel):
    """Schema for individual sentence prediction."""
    
    text: str = Field(..., description="Sentence text")
    label: str = Field(..., description="Predicted label")
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    line_number: int = Field(..., description="Position in abstract", ge=0)
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probabilities for all classes"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This study examined the effects of a new drug.",
                "label": "OBJECTIVE",
                "confidence": 0.92,
                "line_number": 0,
                "probabilities": {
                    "BACKGROUND": 0.05,
                    "OBJECTIVE": 0.92,
                    "METHODS": 0.02,
                    "RESULTS": 0.01,
                    "CONCLUSIONS": 0.00
                }
            }
        }


class AbstractResponse(BaseModel):
    """Response schema for abstract analysis."""
    
    sentences: List[SentencePrediction] = Field(
        ...,
        description="List of analyzed sentences"
    )
    total_sentences: int = Field(..., description="Total number of sentences")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "sentences": [
                    {
                        "text": "This study examined the effects of a new drug.",
                        "label": "OBJECTIVE",
                        "confidence": 0.92,
                        "line_number": 0
                    }
                ],
                "total_sentences": 5,
                "processing_time": 0.045
            }
        }


class BatchAbstractRequest(BaseModel):
    """Request schema for batch processing."""
    
    abstracts: List[AbstractRequest] = Field(
        ...,
        description="List of abstracts to process",
        max_items=100
    )
    
    class Config:
        schema_extra = {
            "example": {
                "abstracts": [
                    {
                        "text": "First abstract text here...",
                        "return_probabilities": False
                    },
                    {
                        "text": "Second abstract text here...",
                        "return_probabilities": True
                    }
                ]
            }
        }


class BatchAbstractResponse(BaseModel):
    """Response schema for batch processing."""
    
    results: List[AbstractResponse] = Field(..., description="List of results")
    total_abstracts: int = Field(..., description="Total abstracts processed")
    total_processing_time: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "2.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not loaded",
                "detail": "The model failed to load on startup"
            }
        }


class StatsResponse(BaseModel):
    """Response schema for API statistics."""
    
    total_requests: int = Field(..., description="Total requests processed")
    total_sentences: int = Field(..., description="Total sentences analyzed")
    avg_processing_time: float = Field(..., description="Average processing time")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 1247,
                "total_sentences": 8934,
                "avg_processing_time": 0.052,
                "uptime_seconds": 86400
            }
        }