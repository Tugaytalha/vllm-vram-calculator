# Contributing to vLLM VRAM Calculator

Thank you for considering contributing to the vLLM VRAM Calculator! This document outlines how to contribute effectively.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Adding a New Model](#adding-a-new-model)
- [Adding a New GPU](#adding-a-new-gpu)
- [Development Guidelines](#development-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/vllm-vram-calculator.git
   cd vllm-vram-calculator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include your Python version, OS, and browser (if UI related)
- Provide steps to reproduce the issue
- Include expected vs actual behavior

### Requesting Features

- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and why it would be valuable
- Consider if you'd be willing to implement it

### Submitting Code

1. Create a branch from `main`
2. Make your changes
3. Add/update tests as needed
4. Update documentation if needed
5. Submit a pull request

## Adding a New Model

To add a new LLM model, edit `api/data/models.json`:

```json
{
  "id": "model-name-size",
  "name": "Model Display Name",
  "provider": "Provider Name",
  "parameters": 7000000000,
  "architecture": {
    "type": "dense",
    "num_layers": 32,
    "hidden_size": 4096,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "attention_type": "gqa"
  }
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (lowercase with hyphens) |
| `name` | Display name |
| `provider` | Company/organization |
| `parameters` | Total parameter count |
| `architecture.type` | `dense` or `moe` |
| `architecture.num_layers` | Number of transformer layers |
| `architecture.hidden_size` | Hidden dimension size |
| `architecture.num_heads` | Number of attention heads |
| `architecture.num_kv_heads` | Number of KV heads (for GQA) |
| `architecture.head_dim` | Dimension per head |
| `architecture.intermediate_size` | FFN intermediate size |
| `architecture.vocab_size` | Vocabulary size |
| `architecture.max_position_embeddings` | Max context length |
| `architecture.attention_type` | `mha`, `gqa`, or `mla` |

### For MoE Models

Add these additional fields:
```json
{
  "active_parameters": 12900000000,
  "architecture": {
    "type": "moe",
    "num_experts": 8,
    "num_experts_per_token": 2
  }
}
```

### Finding Model Parameters

- Check the model's `config.json` on Hugging Face
- Look for the model card or technical report
- Common sources: Hugging Face, GitHub, arXiv papers

## Adding a New GPU

To add a new GPU, edit `api/data/gpus.json`:

```json
{
  "id": "gpu-model-vram",
  "name": "GPU Display Name",
  "vendor": "NVIDIA",
  "vram_gb": 24,
  "memory_type": "GDDR6X",
  "memory_bandwidth_gbps": 1008,
  "tier": "prosumer"
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `name` | Full GPU name |
| `vendor` | NVIDIA, AMD, Apple, etc. |
| `vram_gb` | VRAM in GB |
| `memory_type` | GDDR6, HBM3, etc. |
| `memory_bandwidth_gbps` | Memory bandwidth |
| `tier` | `consumer`, `prosumer`, `datacenter`, or `custom` |

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and returns
- Write docstrings for all public functions
- Use meaningful variable names

### Testing

- Write tests for new features
- Ensure existing tests pass
- Test edge cases (0 values, max values, invalid inputs)

### Documentation

- Update README if adding features
- Add inline comments for complex logic
- Update API docs if changing endpoints

## Pull Request Process

1. **Create a descriptive PR title**
   - Format: `[Type] Brief description`
   - Types: `feat`, `fix`, `docs`, `refactor`, `test`

2. **Fill out the PR template**
   - Describe what changed and why
   - Link related issues
   - List testing done

3. **Ensure CI passes**
   - All tests pass
   - No linting errors

4. **Request review**
   - Tag maintainers for review
   - Address feedback promptly

5. **Squash and merge**
   - Keep commit history clean
   - Use conventional commit messages

## Questions?

- Open a [GitHub Discussion](https://github.com/Tugaytalha/vllm-vram-calculator/discussions)
- Check existing [Issues](https://github.com/Tugaytalha/vllm-vram-calculator/issues)

Thank you for contributing! 🎉
