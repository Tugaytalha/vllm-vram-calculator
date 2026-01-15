# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-15

### Added
- Initial release of vLLM VRAM Calculator
- Core VRAM calculation engine with support for:
  - Model weights memory
  - KV cache memory (with GQA/MHA/MLA support)
  - Activation memory estimation
  - Framework overhead
- REST API with FastAPI:
  - `/api/calculate` - VRAM calculation endpoint
  - `/api/models` - List available models
  - `/api/gpus` - List available GPUs
  - `/api/health` - Health check
- Modern web interface:
  - Dark theme design
  - Real-time calculation updates
  - Memory allocation visualization
  - Responsive layout
- Model database with 26 popular LLMs:
  - Llama 3.x (1B, 3B, 8B, 70B)
  - Mistral 7B, Mixtral 8x7B
  - Qwen 2.5 (0.5B - 72B)
  - DeepSeek-R1 (1.5B - 671B)
  - Gemma 2 (2B, 9B, 27B)
  - Phi-3 (Mini, Small, Medium)
- GPU database with 32 GPUs:
  - NVIDIA RTX 30xx/40xx/50xx series
  - NVIDIA datacenter (A10, A40, L40, A100, H100, H200)
  - AMD MI250X, MI300X
  - Apple Silicon M1/M2/M3/M4
- Quantization support:
  - Weight: FP32, FP16, BF16, INT8, INT4, AWQ, GPTQ
  - KV Cache: FP32, FP16, BF16, INT8, FP8
- Multi-GPU support with tensor parallelism
- Concurrent user scaling calculations
- Project documentation (README, CONTRIBUTING)

[Unreleased]: https://github.com/Tugaytalha/vllm-vram-calculator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Tugaytalha/vllm-vram-calculator/releases/tag/v0.1.0
