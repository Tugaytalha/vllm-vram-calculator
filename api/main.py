"""
vLLM VRAM Calculator API

FastAPI application providing REST endpoints for VRAM calculation.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models import (
    CalculationRequest,
    CalculationResponse,
    MemoryBreakdown,
    ModelInfo,
    GPUInfo,
    ModelsResponse,
    GPUsResponse,
    HealthResponse,
    QuantizationType as APIQuantizationType,
    KVCacheQuantization as APIKVCacheQuantization,
)
from .calculator import (
    CalculationConfig,
    ModelArchitecture,
    VRAMBreakdown,
    calculate_vram,
    calculate_max_concurrent_users,
    QuantizationType,
    KVCacheQuantization,
)

# Initialize FastAPI app
app = FastAPI(
    title="vLLM VRAM Calculator API",
    description="Calculate VRAM requirements for LLM deployments with vLLM",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data files
DATA_DIR = Path(__file__).parent / "data"


def load_models_db() -> dict:
    """Load models database from JSON file."""
    with open(DATA_DIR / "models.json", "r") as f:
        return json.load(f)


def load_gpus_db() -> dict:
    """Load GPUs database from JSON file."""
    with open(DATA_DIR / "gpus.json", "r") as f:
        return json.load(f)


# Cache the databases
MODELS_DB = load_models_db()
GPUS_DB = load_gpus_db()


def get_model_by_id(model_id: str) -> Optional[dict]:
    """Get model data by ID."""
    for model in MODELS_DB["models"]:
        if model["id"] == model_id:
            return model
    return None


def get_gpu_by_id(gpu_id: str) -> Optional[dict]:
    """Get GPU data by ID."""
    for gpu in GPUS_DB["gpus"]:
        if gpu["id"] == gpu_id:
            return gpu
    return None


def map_quantization(api_quant: APIQuantizationType) -> QuantizationType:
    """Map API quantization type to calculator quantization type."""
    return QuantizationType(api_quant.value)


def map_kv_quantization(api_quant: APIKVCacheQuantization) -> KVCacheQuantization:
    """Map API KV quantization type to calculator KV quantization type."""
    return KVCacheQuantization(api_quant.value)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/api/models", response_model=ModelsResponse, tags=["Data"])
async def list_models():
    """List all available LLM models."""
    models = []
    for model in MODELS_DB["models"]:
        arch = model["architecture"]
        models.append(ModelInfo(
            id=model["id"],
            name=model["name"],
            provider=model["provider"],
            parameters=model["parameters"],
            architecture_type=arch["type"],
            num_layers=arch["num_layers"],
            hidden_size=arch["hidden_size"],
            num_heads=arch["num_heads"],
            num_kv_heads=arch["num_kv_heads"],
            max_position_embeddings=arch["max_position_embeddings"],
            attention_type=arch["attention_type"],
        ))
    return ModelsResponse(models=models, count=len(models))


@app.get("/api/gpus", response_model=GPUsResponse, tags=["Data"])
async def list_gpus():
    """List all available GPUs."""
    gpus = []
    for gpu in GPUS_DB["gpus"]:
        gpus.append(GPUInfo(
            id=gpu["id"],
            name=gpu["name"],
            vendor=gpu["vendor"],
            vram_gb=gpu["vram_gb"],
            memory_type=gpu["memory_type"],
            tier=gpu["tier"],
        ))
    return GPUsResponse(gpus=gpus, count=len(gpus))


@app.post("/api/calculate", response_model=CalculationResponse, tags=["Calculation"])
async def calculate(request: CalculationRequest):
    """
    Calculate VRAM requirements for the given configuration.
    
    This endpoint calculates:
    - Total VRAM required
    - Memory breakdown (weights, KV cache, activations, overhead)
    - Whether the configuration fits in the selected GPU
    - Maximum concurrent users supported
    """
    # Get model data
    model = get_model_by_id(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    
    # Get GPU data
    gpu = get_gpu_by_id(request.gpu_id)
    if not gpu:
        raise HTTPException(status_code=404, detail=f"GPU '{request.gpu_id}' not found")
    
    # Handle custom VRAM
    gpu_vram = request.custom_vram_gb if request.gpu_id == "custom" and request.custom_vram_gb else gpu["vram_gb"]
    
    # Build architecture config
    arch = model["architecture"]
    architecture = ModelArchitecture(
        num_layers=arch["num_layers"],
        hidden_size=arch["hidden_size"],
        num_heads=arch["num_heads"],
        num_kv_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        intermediate_size=arch["intermediate_size"],
        vocab_size=arch["vocab_size"],
        max_position_embeddings=arch["max_position_embeddings"],
        attention_type=arch["attention_type"],
        num_experts=arch.get("num_experts"),
        num_experts_per_token=arch.get("num_experts_per_token"),
    )
    
    # Build calculation config
    config = CalculationConfig(
        model_parameters=model["parameters"],
        architecture=architecture,
        weight_quantization=map_quantization(request.quantization),
        kv_cache_quantization=map_kv_quantization(request.kv_cache_quantization),
        batch_size=request.batch_size,
        sequence_length=request.sequence_length,
        concurrent_users=request.concurrent_users,
        num_gpus=request.num_gpus,
        active_parameters=model.get("active_parameters"),
    )
    
    # Calculate VRAM
    result = calculate_vram(config)
    
    # Calculate max concurrent users
    max_users = calculate_max_concurrent_users(config, gpu_vram * request.num_gpus)
    
    # Check if it fits
    fits = result.total_gb <= gpu_vram
    utilization = min((result.total_gb / gpu_vram) * 100, 100) if gpu_vram > 0 else 0
    
    return CalculationResponse(
        memory=MemoryBreakdown(
            weights_gb=result.weights_gb,
            kv_cache_gb=result.kv_cache_gb,
            activations_gb=result.activations_gb,
            overhead_gb=result.overhead_gb,
            total_gb=result.total_gb,
        ),
        gpu_vram_gb=gpu_vram,
        fits_in_memory=fits,
        vram_utilization_percent=round(utilization, 1),
        shared_memory_gb=result.shared_gb,
        per_user_memory_gb=result.per_user_gb,
        kv_cache_per_token_bytes=result.kv_cache_per_token_bytes,
        max_tokens_in_flight=result.max_tokens_in_flight,
        max_concurrent_users=max_users,
        model_name=model["name"],
        gpu_name=gpu["name"],
    )


# Serve static files for the web interface
WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    
    @app.get("/", tags=["Web"])
    async def serve_index():
        """Serve the web interface."""
        return FileResponse(WEB_DIR / "index.html")
