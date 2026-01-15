"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_returns_ok(self):
        """Health check should return healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestModelsEndpoint:
    """Tests for the models listing endpoint."""
    
    def test_list_models(self):
        """Should return list of available models."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert data["count"] > 0
        assert len(data["models"]) == data["count"]
    
    def test_model_has_required_fields(self):
        """Each model should have required fields."""
        response = client.get("/api/models")
        data = response.json()
        model = data["models"][0]
        
        required_fields = ["id", "name", "provider", "parameters", 
                          "num_layers", "hidden_size", "num_heads"]
        for field in required_fields:
            assert field in model


class TestGPUsEndpoint:
    """Tests for the GPUs listing endpoint."""
    
    def test_list_gpus(self):
        """Should return list of available GPUs."""
        response = client.get("/api/gpus")
        assert response.status_code == 200
        data = response.json()
        assert "gpus" in data
        assert "count" in data
        assert data["count"] > 0
    
    def test_gpu_has_required_fields(self):
        """Each GPU should have required fields."""
        response = client.get("/api/gpus")
        data = response.json()
        gpu = data["gpus"][0]
        
        required_fields = ["id", "name", "vendor", "vram_gb"]
        for field in required_fields:
            assert field in gpu


class TestCalculateEndpoint:
    """Tests for the VRAM calculation endpoint."""
    
    def test_calculate_valid_request(self):
        """Valid calculation request should succeed."""
        response = client.post("/api/calculate", json={
            "model_id": "llama-3.1-8b",
            "gpu_id": "rtx-4090-24gb",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "memory" in data
        assert "weights_gb" in data["memory"]
        assert "kv_cache_gb" in data["memory"]
        assert "total_gb" in data["memory"]
        assert "fits_in_memory" in data
        assert "vram_utilization_percent" in data
    
    def test_calculate_invalid_model(self):
        """Invalid model ID should return 404."""
        response = client.post("/api/calculate", json={
            "model_id": "nonexistent-model",
            "gpu_id": "rtx-4090-24gb",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        assert response.status_code == 404
    
    def test_calculate_invalid_gpu(self):
        """Invalid GPU ID should return 404."""
        response = client.post("/api/calculate", json={
            "model_id": "llama-3.1-8b",
            "gpu_id": "nonexistent-gpu",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        assert response.status_code == 404
    
    def test_calculate_llama_70b_exceeds_rtx_3060(self):
        """Llama 70B should not fit in RTX 3060."""
        response = client.post("/api/calculate", json={
            "model_id": "llama-3.3-70b",
            "gpu_id": "rtx-3060-12gb",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["fits_in_memory"] == False
        # Utilization is capped at 100% in the API
        assert data["vram_utilization_percent"] == 100
    
    def test_calculate_small_model_fits(self):
        """Small model should fit in large GPU."""
        response = client.post("/api/calculate", json={
            "model_id": "qwen2.5-0.5b",
            "gpu_id": "rtx-4090-24gb",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["fits_in_memory"] == True
        assert data["vram_utilization_percent"] < 100
    
    def test_quantization_reduces_memory(self):
        """INT4 should use less memory than FP16."""
        fp16_response = client.post("/api/calculate", json={
            "model_id": "llama-3.1-8b",
            "gpu_id": "rtx-4090-24gb",
            "quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        
        int4_response = client.post("/api/calculate", json={
            "model_id": "llama-3.1-8b",
            "gpu_id": "rtx-4090-24gb",
            "quantization": "int4",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 4096,
            "concurrent_users": 1,
            "num_gpus": 1
        })
        
        fp16_data = fp16_response.json()
        int4_data = int4_response.json()
        
        assert int4_data["memory"]["weights_gb"] < fp16_data["memory"]["weights_gb"]
        assert int4_data["memory"]["total_gb"] < fp16_data["memory"]["total_gb"]
