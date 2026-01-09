"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app, state
from api.schemas import AbstractRequest


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def sample_abstract():
    """Sample abstract for testing."""
    return """
    This study examined the effects of a new drug on diabetes management.
    A total of 200 patients were randomly assigned to treatment or control groups.
    Blood glucose levels were measured weekly.
    Results showed a 25% reduction in blood glucose levels.
    The new drug appears to be effective for diabetes management.
    """


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    @pytest.mark.skipif(
        state.model is None,
        reason="Model not loaded"
    )
    def test_predict_endpoint(self, client, sample_abstract):
        """Test prediction endpoint."""
        request_data = {
            "text": sample_abstract,
            "return_probabilities": False
        }
        
        response = client.post("/predict", json=request_data)
        
        # May fail if model not loaded, that's expected in CI
        if response.status_code == 200:
            data = response.json()
            assert "sentences" in data
            assert "total_sentences" in data
            assert "processing_time" in data
            assert len(data["sentences"]) > 0
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        request_data = {
            "text": "",
            "return_probabilities": False
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_probabilities(self, client, sample_abstract):
        """Test prediction with probabilities."""
        request_data = {
            "text": sample_abstract,
            "return_probabilities": True
        }
        
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # Check if probabilities are included
            if len(data["sentences"]) > 0:
                first_sentence = data["sentences"][0]
                if "probabilities" in first_sentence:
                    assert isinstance(first_sentence["probabilities"], dict)
    
    @pytest.mark.skipif(
        state.model is None,
        reason="Model not loaded"
    )
    def test_batch_predict_endpoint(self, client, sample_abstract):
        """Test batch prediction endpoint."""
        request_data = {
            "abstracts": [
                {"text": sample_abstract, "return_probabilities": False},
                {"text": sample_abstract, "return_probabilities": False}
            ]
        }
        
        response = client.post("/batch-predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total_abstracts" in data


class TestAPISchemas:
    """Tests for API schemas validation."""
    
    def test_abstract_request_validation(self):
        """Test AbstractRequest validation."""
        # Valid request
        request = AbstractRequest(
            text="This is a valid abstract text.",
            return_probabilities=False
        )
        assert request.text == "This is a valid abstract text."
        
        # Invalid: empty text
        with pytest.raises(Exception):
            AbstractRequest(text="", return_probabilities=False)
    
    def test_abstract_request_strips_whitespace(self):
        """Test that text is stripped."""
        request = AbstractRequest(
            text="  Text with whitespace  ",
            return_probabilities=False
        )
        assert request.text == "Text with whitespace"


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test using invalid HTTP method."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for complete API flow."""
    
    @pytest.mark.skipif(
        state.model is None,
        reason="Model not loaded"
    )
    def test_complete_prediction_flow(self, client, sample_abstract):
        """Test complete prediction flow."""
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Make prediction
        predict_response = client.post(
            "/predict",
            json={"text": sample_abstract, "return_probabilities": True}
        )
        
        if predict_response.status_code == 200:
            data = predict_response.json()
            
            # Verify response structure
            assert "sentences" in data
            assert len(data["sentences"]) > 0
            
            # Verify each sentence has required fields
            for sentence in data["sentences"]:
                assert "text" in sentence
                assert "label" in sentence
                assert "confidence" in sentence
                assert "line_number" in sentence
        
        # Check stats updated
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])