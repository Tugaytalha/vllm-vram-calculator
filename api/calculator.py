"""
vLLM VRAM Calculator - Core Calculation Engine

This module implements the VRAM calculation formulas based on:
- NVIDIA's LLM inference optimization documentation
- vLLM's memory management approach
- TensorRT-LLM memory usage specifications

Key formulas:
1. Model Weights: M_weights = num_parameters * bytes_per_param
2. KV Cache per token: m_token = 2 * num_layers * H_kv * bytes_per_kv
   where H_kv = num_kv_heads * head_dim
3. Total KV Cache: M_kv_total = total_tokens_in_flight * m_token
4. Framework overhead: ~2-5% of total memory
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantizationType(str, Enum):
    """Supported quantization types for model weights."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"      # 4-bit
    GPTQ = "gptq"    # 4-bit
    GGUF_Q4 = "gguf_q4"
    GGUF_Q8 = "gguf_q8"


class KVCacheQuantization(str, Enum):
    """Supported quantization types for KV cache."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"


# Bytes per element for each quantization type
QUANTIZATION_BYTES = {
    QuantizationType.FP32: 4.0,
    QuantizationType.FP16: 2.0,
    QuantizationType.BF16: 2.0,
    QuantizationType.INT8: 1.0,
    QuantizationType.INT4: 0.5,
    QuantizationType.AWQ: 0.5,
    QuantizationType.GPTQ: 0.5,
    QuantizationType.GGUF_Q4: 0.5,
    QuantizationType.GGUF_Q8: 1.0,
}

KV_CACHE_BYTES = {
    KVCacheQuantization.FP32: 4.0,
    KVCacheQuantization.FP16: 2.0,
    KVCacheQuantization.BF16: 2.0,
    KVCacheQuantization.INT8: 1.0,
    KVCacheQuantization.FP8: 1.0,
}


@dataclass
class ModelArchitecture:
    """Model architecture parameters needed for VRAM calculation."""
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    attention_type: str  # "mha", "gqa", "mla"
    # For MoE models
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None


@dataclass
class CalculationConfig:
    """Configuration parameters for VRAM calculation."""
    # Model configuration
    model_parameters: float  # Total model parameters
    architecture: ModelArchitecture
    
    # Quantization settings
    weight_quantization: QuantizationType = QuantizationType.FP16
    kv_cache_quantization: KVCacheQuantization = KVCacheQuantization.FP16
    
    # Inference parameters
    batch_size: int = 1
    sequence_length: int = 2048
    concurrent_users: int = 1
    
    # GPU configuration
    num_gpus: int = 1
    
    # For MoE models - use active parameters instead of total
    active_parameters: Optional[float] = None


@dataclass
class VRAMBreakdown:
    """Detailed breakdown of VRAM usage."""
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    
    # Per-user breakdown
    shared_gb: float  # Weights + base activations
    per_user_gb: float  # KV cache per user
    
    # Additional metrics
    kv_cache_per_token_bytes: float
    max_tokens_in_flight: int


def calculate_weight_memory(
    num_parameters: float,
    quantization: QuantizationType,
    active_parameters: Optional[float] = None
) -> float:
    """
    Calculate memory required for model weights.
    
    For MoE models, use active_parameters if provided.
    
    Args:
        num_parameters: Total number of model parameters
        quantization: Weight quantization type
        active_parameters: For MoE models, the number of active parameters per token
        
    Returns:
        Memory in GB
    """
    # For MoE models, we still need to load all weights
    # But active_parameters affects activation memory
    params = num_parameters
    bytes_per_param = QUANTIZATION_BYTES[quantization]
    
    memory_bytes = params * bytes_per_param
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb


def calculate_kv_cache_per_token(
    architecture: ModelArchitecture,
    kv_quantization: KVCacheQuantization
) -> float:
    """
    Calculate KV cache memory per token.
    
    Formula: m_token = 2 * num_layers * H_kv * bytes_per_kv
    where H_kv = num_kv_heads * head_dim
    
    The factor of 2 accounts for both K and V caches.
    
    Args:
        architecture: Model architecture parameters
        kv_quantization: KV cache quantization type
        
    Returns:
        Memory in bytes per token
    """
    bytes_per_kv = KV_CACHE_BYTES[kv_quantization]
    
    # H_kv = num_kv_heads * head_dim
    h_kv = architecture.num_kv_heads * architecture.head_dim
    
    # m_token = 2 * L * H_kv * b (factor of 2 for K and V)
    memory_per_token = 2 * architecture.num_layers * h_kv * bytes_per_kv
    
    return memory_per_token


def calculate_total_kv_cache(
    architecture: ModelArchitecture,
    kv_quantization: KVCacheQuantization,
    batch_size: int,
    sequence_length: int,
    concurrent_users: int
) -> tuple[float, int]:
    """
    Calculate total KV cache memory for all concurrent requests.
    
    Formula: M_kv_total = sum(S_i) * m_token
    where S_i is the sequence length for each active request.
    
    For simplified case where all requests have same sequence length:
    M_kv_total = N * S * m_token
    
    Args:
        architecture: Model architecture parameters
        kv_quantization: KV cache quantization type
        batch_size: Batch size per user
        sequence_length: Maximum sequence length
        concurrent_users: Number of concurrent users/requests
        
    Returns:
        Tuple of (Memory in GB, total tokens in flight)
    """
    bytes_per_token = calculate_kv_cache_per_token(architecture, kv_quantization)
    
    # Total tokens in flight = concurrent_users * batch_size * sequence_length
    total_tokens = concurrent_users * batch_size * sequence_length
    
    total_memory_bytes = total_tokens * bytes_per_token
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    return total_memory_gb, total_tokens


def calculate_activation_memory(
    architecture: ModelArchitecture,
    batch_size: int,
    sequence_length: int,
    weight_quantization: QuantizationType,
    active_parameters: Optional[float] = None
) -> float:
    """
    Calculate activation memory during inference.
    
    Activation memory is typically proportional to:
    - Batch size
    - Sequence length  
    - Hidden size
    - Number of layers (for intermediate activations)
    
    This is an approximation as actual activation memory depends on
    specific implementation details.
    
    Args:
        architecture: Model architecture parameters
        batch_size: Current batch size
        sequence_length: Current sequence length
        weight_quantization: Weight quantization (affects activation precision)
        active_parameters: For MoE, number of active parameters
        
    Returns:
        Memory in GB
    """
    # Activation precision matches weight precision for compute
    # but intermediate activations are often kept in higher precision
    if weight_quantization in [QuantizationType.FP32]:
        bytes_per_activation = 4.0
    else:
        bytes_per_activation = 2.0  # FP16/BF16 for activations
    
    # Rough estimation of activation memory:
    # - Input/output embeddings: batch * seq * hidden
    # - Attention scores: batch * num_heads * seq * seq (for self-attention)
    # - FFN intermediate: batch * seq * intermediate_size
    
    # Simplified approximation
    hidden_size = architecture.hidden_size
    intermediate_size = architecture.intermediate_size
    num_heads = architecture.num_heads
    
    # Per-layer activation memory
    # Attention: Q, K, V projections + attention output
    attention_mem = batch_size * sequence_length * hidden_size * 4 * bytes_per_activation
    
    # FFN: intermediate activations (only need to store one at a time during inference)
    ffn_mem = batch_size * sequence_length * intermediate_size * bytes_per_activation
    
    # Total activation memory (we don't need all layers simultaneously during inference)
    # In inference, we process layer by layer, so we only need memory for ~2 layers
    activation_bytes = (attention_mem + ffn_mem) * 2
    
    activation_gb = activation_bytes / (1024 ** 3)
    
    return activation_gb


def calculate_overhead(
    weights_gb: float,
    kv_cache_gb: float,
    activations_gb: float,
    overhead_factor: float = 0.05
) -> float:
    """
    Calculate framework overhead.
    
    vLLM and other serving frameworks have overhead for:
    - CUDA context
    - Memory fragmentation
    - Internal buffers
    - Page tables for paged attention
    
    Typically 2-5% of total memory.
    
    Args:
        weights_gb: Weight memory in GB
        kv_cache_gb: KV cache memory in GB
        activations_gb: Activation memory in GB
        overhead_factor: Overhead as fraction of total (default 5%)
        
    Returns:
        Overhead memory in GB
    """
    base_memory = weights_gb + kv_cache_gb + activations_gb
    overhead = base_memory * overhead_factor
    
    # Minimum overhead of ~500MB for CUDA context etc.
    min_overhead = 0.5
    
    return max(overhead, min_overhead)


def calculate_vram(config: CalculationConfig) -> VRAMBreakdown:
    """
    Calculate total VRAM required for the given configuration.
    
    This is the main entry point for VRAM calculation.
    
    Args:
        config: Complete calculation configuration
        
    Returns:
        VRAMBreakdown with detailed memory usage
    """
    # 1. Calculate weight memory
    weights_gb = calculate_weight_memory(
        config.model_parameters,
        config.weight_quantization,
        config.active_parameters
    )
    
    # 2. Calculate KV cache memory
    kv_cache_gb, total_tokens = calculate_total_kv_cache(
        config.architecture,
        config.kv_cache_quantization,
        config.batch_size,
        config.sequence_length,
        config.concurrent_users
    )
    
    # 3. Calculate activation memory
    activations_gb = calculate_activation_memory(
        config.architecture,
        config.batch_size,
        config.sequence_length,
        config.weight_quantization,
        config.active_parameters
    )
    
    # 4. Calculate overhead
    overhead_gb = calculate_overhead(weights_gb, kv_cache_gb, activations_gb)
    
    # 5. Total VRAM
    total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb
    
    # 6. Distribute across GPUs if using tensor parallelism
    if config.num_gpus > 1:
        # Weights are split across GPUs
        weights_gb = weights_gb / config.num_gpus
        # KV cache is also split
        kv_cache_gb = kv_cache_gb / config.num_gpus
        # Activations are split
        activations_gb = activations_gb / config.num_gpus
        # Overhead per GPU remains similar (each GPU has its own context)
        # But total overhead doesn't scale linearly
        overhead_gb = overhead_gb / config.num_gpus + 0.3  # Base overhead per GPU
        
        total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb
    
    # Calculate per-token KV cache
    kv_per_token = calculate_kv_cache_per_token(
        config.architecture, 
        config.kv_cache_quantization
    )
    
    # Shared memory (weights + base activations + overhead)
    shared_gb = weights_gb + activations_gb + overhead_gb
    
    # Per-user memory (KV cache per user)
    per_user_kv_gb = kv_cache_gb / max(config.concurrent_users, 1)
    
    return VRAMBreakdown(
        weights_gb=round(weights_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        activations_gb=round(activations_gb, 2),
        overhead_gb=round(overhead_gb, 2),
        total_gb=round(total_gb, 2),
        shared_gb=round(shared_gb, 2),
        per_user_gb=round(per_user_kv_gb, 2),
        kv_cache_per_token_bytes=round(kv_per_token, 2),
        max_tokens_in_flight=total_tokens
    )


def calculate_max_concurrent_users(
    config: CalculationConfig,
    available_vram_gb: float
) -> int:
    """
    Calculate maximum number of concurrent users given available VRAM.
    
    Uses the formula:
    T_max = floor((M_VRAM - M_weights - M_activations - M_overhead) / m_token)
    N_max = floor(T_max / sequence_length)
    
    Args:
        config: Calculation configuration
        available_vram_gb: Available VRAM in GB
        
    Returns:
        Maximum number of concurrent users
    """
    # Calculate fixed memory costs
    weights_gb = calculate_weight_memory(
        config.model_parameters,
        config.weight_quantization
    )
    
    activations_gb = calculate_activation_memory(
        config.architecture,
        config.batch_size,
        config.sequence_length,
        config.weight_quantization
    )
    
    # Estimate overhead
    overhead_gb = calculate_overhead(weights_gb, 0, activations_gb)
    
    # Available for KV cache
    kv_budget_gb = available_vram_gb - weights_gb - activations_gb - overhead_gb
    
    if kv_budget_gb <= 0:
        return 0
    
    # KV cache per token
    kv_per_token = calculate_kv_cache_per_token(
        config.architecture,
        config.kv_cache_quantization
    )
    
    # Convert budget to bytes
    kv_budget_bytes = kv_budget_gb * (1024 ** 3)
    
    # Max tokens
    max_tokens = int(kv_budget_bytes / kv_per_token)
    
    # Max concurrent users
    tokens_per_user = config.batch_size * config.sequence_length
    max_users = max_tokens // tokens_per_user
    
    return max(max_users, 0)
