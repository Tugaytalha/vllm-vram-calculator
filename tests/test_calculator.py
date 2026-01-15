"""
Tests for the VRAM calculation engine.
"""

import pytest
from api.calculator import (
    calculate_vram,
    calculate_kv_cache_per_token,
    calculate_weight_memory,
    calculate_max_concurrent_users,
    CalculationConfig,
    ModelArchitecture,
    QuantizationType,
    KVCacheQuantization,
)


# Test fixtures
@pytest.fixture
def llama_8b_architecture():
    """Llama 3.1 8B architecture."""
    return ModelArchitecture(
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
        attention_type="gqa"
    )


@pytest.fixture
def llama_70b_architecture():
    """Llama 3.3 70B architecture."""
    return ModelArchitecture(
        num_layers=80,
        hidden_size=8192,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=28672,
        vocab_size=128256,
        max_position_embeddings=131072,
        attention_type="gqa"
    )


class TestWeightMemory:
    """Tests for weight memory calculation."""
    
    def test_fp16_weight_memory(self):
        """Test FP16 weight memory calculation."""
        # 8B parameters at FP16 (2 bytes) = 16 GB
        memory = calculate_weight_memory(8e9, QuantizationType.FP16)
        assert 14.5 < memory < 15.5  # ~14.9 GB with some overhead
    
    def test_int4_weight_memory(self):
        """Test INT4 weight memory calculation."""
        # 8B parameters at INT4 (0.5 bytes) = 4 GB
        memory = calculate_weight_memory(8e9, QuantizationType.INT4)
        assert 3.5 < memory < 4.5
    
    def test_fp32_weight_memory(self):
        """Test FP32 weight memory calculation."""
        # 8B parameters at FP32 (4 bytes) = 32 GB
        memory = calculate_weight_memory(8e9, QuantizationType.FP32)
        assert 28 < memory < 32


class TestKVCache:
    """Tests for KV cache calculation."""
    
    def test_kv_cache_per_token_llama_8b(self, llama_8b_architecture):
        """Test KV cache per token for Llama 8B with GQA."""
        # Formula: 2 * num_layers * num_kv_heads * head_dim * bytes
        # = 2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KB per token
        bytes_per_token = calculate_kv_cache_per_token(
            llama_8b_architecture,
            KVCacheQuantization.FP16
        )
        assert 125000 < bytes_per_token < 135000  # ~131072 bytes
    
    def test_kv_cache_per_token_llama_70b(self, llama_70b_architecture):
        """Test KV cache per token for Llama 70B with GQA."""
        # Formula: 2 * 80 * 8 * 128 * 2 = 327680 bytes = 320 KB per token
        bytes_per_token = calculate_kv_cache_per_token(
            llama_70b_architecture,
            KVCacheQuantization.FP16
        )
        assert 320000 < bytes_per_token < 340000  # ~327680 bytes
    
    def test_kv_cache_int8_half_of_fp16(self, llama_8b_architecture):
        """INT8 KV cache should be half the size of FP16."""
        fp16_bytes = calculate_kv_cache_per_token(
            llama_8b_architecture,
            KVCacheQuantization.FP16
        )
        int8_bytes = calculate_kv_cache_per_token(
            llama_8b_architecture,
            KVCacheQuantization.INT8
        )
        assert abs(int8_bytes - fp16_bytes / 2) < 100


class TestVRAMCalculation:
    """Tests for full VRAM calculation."""
    
    def test_basic_calculation(self, llama_8b_architecture):
        """Test basic VRAM calculation for Llama 8B."""
        config = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=1,
            num_gpus=1
        )
        result = calculate_vram(config)
        
        # Weights should be ~15 GB
        assert 14 < result.weights_gb < 16
        # KV cache at 4096 tokens should be ~0.5 GB
        assert 0.3 < result.kv_cache_gb < 1.0
        # Total should be reasonable
        assert 15 < result.total_gb < 20
    
    def test_concurrent_users_scale_kv_cache(self, llama_8b_architecture):
        """KV cache should scale linearly with concurrent users."""
        config_1_user = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=1,
            num_gpus=1
        )
        
        config_4_users = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=4,
            num_gpus=1
        )
        
        result_1 = calculate_vram(config_1_user)
        result_4 = calculate_vram(config_4_users)
        
        # KV cache should be ~4x for 4 users
        assert 3.5 < result_4.kv_cache_gb / result_1.kv_cache_gb < 4.5
        # Weights should be the same
        assert result_1.weights_gb == result_4.weights_gb
    
    def test_multi_gpu_splits_memory(self, llama_8b_architecture):
        """Multi-GPU should reduce per-GPU memory."""
        config_1_gpu = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=1,
            num_gpus=1
        )
        
        config_2_gpu = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=1,
            num_gpus=2
        )
        
        result_1 = calculate_vram(config_1_gpu)
        result_2 = calculate_vram(config_2_gpu)
        
        # Per-GPU memory should be roughly half (with some overhead per GPU)
        assert result_2.weights_gb < result_1.weights_gb


class TestMaxConcurrentUsers:
    """Tests for max concurrent users calculation."""
    
    def test_max_users_increases_with_vram(self, llama_8b_architecture):
        """More VRAM should allow more concurrent users."""
        config = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=4096,
            concurrent_users=1,
            num_gpus=1
        )
        
        max_users_24gb = calculate_max_concurrent_users(config, 24)
        max_users_48gb = calculate_max_concurrent_users(config, 48)
        
        assert max_users_48gb > max_users_24gb
    
    def test_max_users_decreases_with_sequence_length(self, llama_8b_architecture):
        """Longer sequences should reduce max concurrent users."""
        config_short = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=2048,
            concurrent_users=1,
            num_gpus=1
        )
        
        config_long = CalculationConfig(
            model_parameters=8e9,
            architecture=llama_8b_architecture,
            weight_quantization=QuantizationType.FP16,
            kv_cache_quantization=KVCacheQuantization.FP16,
            batch_size=1,
            sequence_length=8192,
            concurrent_users=1,
            num_gpus=1
        )
        
        max_users_short = calculate_max_concurrent_users(config_short, 24)
        max_users_long = calculate_max_concurrent_users(config_long, 24)
        
        assert max_users_short > max_users_long
