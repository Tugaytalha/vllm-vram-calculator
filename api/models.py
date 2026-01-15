"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class QuantizationType(str, Enum):
    """Supported quantization types for model weights."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q8 = "gguf_q8"


class KVCacheQuantization(str, Enum):
    """Supported quantization types for KV cache."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"


class CalculationRequest(BaseModel):
    """Request model for VRAM calculation."""
    model_id: str = Field(..., description="ID of the model from the models database")
    gpu_id: str = Field(..., description="ID of the GPU from the GPUs database")
    
    # Quantization settings
    quantization: QuantizationType = Field(
        default=QuantizationType.FP16,
        description="Weight quantization type"
    )
    kv_cache_quantization: KVCacheQuantization = Field(
        default=KVCacheQuantization.FP16,
        description="KV cache quantization type"
    )
    
    # Inference parameters
    batch_size: int = Field(default=1, ge=1, le=256, description="Batch size")
    sequence_length: int = Field(
        default=4096, 
        ge=128, 
        le=131072, 
        description="Maximum sequence length in tokens"
    )
    concurrent_users: int = Field(
        default=1, 
        ge=1, 
        le=1000, 
        description="Number of concurrent users/requests"
    )
    
    # GPU configuration
    num_gpus: int = Field(default=1, ge=1, le=8, description="Number of GPUs")
    custom_vram_gb: Optional[float] = Field(
        default=None,
        description="Custom VRAM in GB (used when gpu_id is 'custom')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "llama-3.1-8b",
                "gpu_id": "rtx-4090-24gb",
                "quantization": "fp16",
                "kv_cache_quantization": "fp16",
                "batch_size": 1,
                "sequence_length": 4096,
                "concurrent_users": 1,
                "num_gpus": 1
            }
        }


class MemoryBreakdown(BaseModel):
    """Detailed memory breakdown."""
    weights_gb: float = Field(..., description="Memory for model weights in GB")
    kv_cache_gb: float = Field(..., description="Memory for KV cache in GB")
    activations_gb: float = Field(..., description="Memory for activations in GB")
    overhead_gb: float = Field(..., description="Framework overhead in GB")
    total_gb: float = Field(..., description="Total VRAM required in GB")


class CalculationResponse(BaseModel):
    """Response model for VRAM calculation."""
    # Memory breakdown
    memory: MemoryBreakdown
    
    # GPU comparison
    gpu_vram_gb: float = Field(..., description="Available VRAM on selected GPU")
    fits_in_memory: bool = Field(..., description="Whether the configuration fits in GPU memory")
    vram_utilization_percent: float = Field(..., description="Percentage of GPU VRAM used")
    
    # Per-user breakdown
    shared_memory_gb: float = Field(..., description="Shared memory (weights + activations + overhead)")
    per_user_memory_gb: float = Field(..., description="KV cache memory per user")
    
    # Additional metrics
    kv_cache_per_token_bytes: float = Field(..., description="KV cache bytes per token")
    max_tokens_in_flight: int = Field(..., description="Total tokens that can be in flight")
    max_concurrent_users: int = Field(..., description="Maximum concurrent users for this GPU")
    
    # Input echo for reference
    model_name: str
    gpu_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "memory": {
                    "weights_gb": 16.0,
                    "kv_cache_gb": 2.0,
                    "activations_gb": 0.5,
                    "overhead_gb": 1.0,
                    "total_gb": 19.5
                },
                "gpu_vram_gb": 24.0,
                "fits_in_memory": True,
                "vram_utilization_percent": 81.25,
                "shared_memory_gb": 17.5,
                "per_user_memory_gb": 2.0,
                "kv_cache_per_token_bytes": 524288,
                "max_tokens_in_flight": 4096,
                "max_concurrent_users": 3,
                "model_name": "Llama 3.1 8B",
                "gpu_name": "NVIDIA RTX 4090"
            }
        }


class ModelInfo(BaseModel):
    """Model information from the database."""
    id: str
    name: str
    provider: str
    parameters: float
    architecture_type: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    max_position_embeddings: int
    attention_type: str


class GPUInfo(BaseModel):
    """GPU information from the database."""
    id: str
    name: str
    vendor: str
    vram_gb: float
    memory_type: str
    tier: str


class ModelsResponse(BaseModel):
    """Response for listing all models."""
    models: List[ModelInfo]
    count: int


class GPUsResponse(BaseModel):
    """Response for listing all GPUs."""
    gpus: List[GPUInfo]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"
