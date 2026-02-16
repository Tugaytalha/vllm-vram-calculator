from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_llama4_moe_params():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()["models"]
    
    # Check Llama 4 Scout
    scout = next((m for m in models if "scout" in m["id"] and "llama-4" in m["id"]), None)
    
    # If using restricted list, it might be the only one
    if scout:
        print(f"Scout Params: {scout['parameters']}")
        # Expect ~109B params (102B in our calc)
        assert scout["parameters"] > 90_000_000_000, f"Scout params {scout['parameters']} too low"
        
        # Calculate VRAM
        req = {
            "model_id": scout["id"],
            "gpu_id": "rtx-4090-24gb",
            "weight_quantization": "fp16",
            "kv_cache_quantization": "fp16",
            "batch_size": 1,
            "sequence_length": 1,
            "concurrent_users": 1,
            "num_gpus": 1
        }
        res = client.post("/api/calculate", json=req)
        assert res.status_code == 200
        data = res.json()
        weights = data["memory"]["weights_gb"]
        print(f"Scout Weights: {weights} GB")
        # 102B * 2 bytes = 204GB / 1024^3 ~ 190 GB
        assert weights > 180, f"Weights {weights}GB too low"

def test_phi3_context():
    response = client.get("/api/models")
    models = response.json()["models"]
    phi = next((m for m in models if "phi-3-mini-4k" in m["id"]), None)
    if phi:
        print(f"Phi-3 Context: {phi['architecture']['max_position_embeddings']}")
        assert phi["architecture"]["max_position_embeddings"] == 4096
