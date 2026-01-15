# vLLM VRAM Calculator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**Precise GPU memory estimation for LLM deployments with vLLM**

Calculate model weights, KV cache requirements, and activation memory across various precisions (FP16, INT8, INT4, AWQ, GPTQ) and multi-GPU setups. Features a clean, interactive Web Interface and a REST API for seamless MLOps integration.

![VRAM Calculator Screenshot](docs/screenshot.png)

## ✨ Features

- **Accurate VRAM Estimation**: Based on NVIDIA's LLM inference optimization formulas
- **Multiple Quantization Options**: FP32, FP16, BF16, INT8, INT4, AWQ, GPTQ
- **KV Cache Calculation**: Precise KV cache sizing with GQA/MHA support
- **Concurrent User Scaling**: See how VRAM scales with parallel requests
- **Multi-GPU Support**: Tensor parallelism memory distribution
- **Modern Web Interface**: Dark theme, responsive design, real-time updates
- **REST API**: Integrate into your MLOps pipelines
- **Extensible Model Database**: Easily add new models via JSON

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Tugaytalha/vllm-vram-calculator.git
cd vllm-vram-calculator

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser to use the calculator.

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-3.1-8b",
    "gpu_id": "rtx-4090-24gb",
    "quantization": "fp16",
    "kv_cache_quantization": "fp16",
    "batch_size": 1,
    "sequence_length": 4096,
    "concurrent_users": 1,
    "num_gpus": 1
  }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calculate` | POST | Calculate VRAM for configuration |
| `/api/models` | GET | List available LLM models |
| `/api/gpus` | GET | List available GPUs |
| `/api/health` | GET | Health check |

## 🧮 Calculation Methodology

### Model Weights
```
M_weights = num_parameters × bytes_per_param
```

### KV Cache (per token)
```
m_token = 2 × num_layers × H_kv × bytes_per_kv
```
Where `H_kv = num_kv_heads × head_dim` (factor of 2 accounts for K and V)

### Total KV Cache (concurrent requests)
```
M_kv_total = Σ(sequence_lengths) × m_token
```

### Total VRAM
```
M_total = M_weights + M_kv_cache + M_activations + M_overhead
```

For detailed formulas, see [NVIDIA's LLM Inference Optimization Guide](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/).

## 🗂️ Supported Models

| Provider | Models |
|----------|--------|
| Meta | Llama 3.2 (1B, 3B), Llama 3.1 8B, Llama 3.3 70B |
| Mistral AI | Mistral 7B, Mixtral 8x7B |
| Alibaba | Qwen 2.5 (0.5B - 72B) |
| DeepSeek | DeepSeek-R1 (1.5B - 671B) |
| Google | Gemma 2 (2B, 9B, 27B) |
| Microsoft | Phi-3 (Mini, Small, Medium) |

[Add a new model →](CONTRIBUTING.md#adding-a-new-model)

## 🖥️ Supported GPUs

**NVIDIA Consumer**: RTX 3060-3090, RTX 4060-4090, RTX 5090  
**NVIDIA Datacenter**: A10, A40, L40/L40S, A100 (40/80GB), H100, H200  
**AMD**: MI250X, MI300X  
**Apple Silicon**: M1/M2/M3/M4 (various configurations)

[Add a new GPU →](CONTRIBUTING.md#adding-a-new-gpu)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to report bugs and request features
- Development setup guide
- Adding new models and GPUs
- Code style guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for the excellent LLM serving framework
- [NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) for LLM optimization documentation
- [Hugging Face](https://huggingface.co/) for model specifications

## 📞 Contact

- **Author**: Tugay Talha İçen
- **GitHub**: [@Tugaytalha](https://github.com/Tugaytalha)
- **Issues**: [GitHub Issues](https://github.com/Tugaytalha/vllm-vram-calculator/issues)

---

⭐ Star this repo if you find it useful!
