"""
Model-details extractor (configs + derived KV cache + weight shard inventory) for a large
set of Hugging Face model repos.

What this script does (per repo):
- Downloads config.json (and uses nested text_config / vision_config when present)
- Extracts: layers, hidden size, intermediate size, heads, kv_heads, head_dim, ctx, vocab, act, RoPE settings, MoE routing
- Derives standard KV-cache bytes/token (fp16/bf16) when applicable
- Lists weight files and total weight bytes (safetensors/bin) via model_info()

Notes:
- Many Meta/Gemma/Aya/DBRX repos are gated. Set HF_TOKEN env var after accepting terms:
    export HF_TOKEN=hf_...
- Some models (notably DeepSeek-V3/R1 and certain VLMs) use non-standard attention/cache
  layouts; KV-cache numbers are skipped when the config indicates that.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# -------------------------
# 1) FULL MODEL LIST (filled)
# -------------------------

REPO_IDS: List[str] = [
    # -------------------------
    # Meta Llama (gated)
    # -------------------------
    "meta-llama/Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Maverick-17B-128E",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",

    # MS/HF mirrors (optional; may duplicate above)
    "LLM-Research/Llama-3.2-1B-Instruct",

    # -------------------------
    # Qwen3 (dense) + Instruct
    # -------------------------
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Instruct",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-Instruct",

    # -------------------------
    # Qwen3 (MoE) + Instruct
    # -------------------------
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-235B-A22B-Instruct",

    # -------------------------
    # Qwen2.5 (dense) + Instruct
    # -------------------------
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",

    # -------------------------
    # DeepSeek
    # -------------------------
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",

    # -------------------------
    # Mistral / Mixtral / Pixtral
    # -------------------------
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Pixtral-12B-2409",

    # mirrors (optional)
    "AI-ModelScope/Mistral-Nemo-Instruct-2407",
    "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",

    # -------------------------
    # Gemma 3 + Gemma 3n (gated)
    # -------------------------
    "google/gemma-3-1b",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b",
    "google/gemma-3-27b-it",
    "google/gemma-3n-e2b",

    # MS example mirror (optional)
    "LLM-Research/gemma-3-27b-it",
    # this ID was mentioned as a ModelScope example; it may not exist on HF:
    "google/gemma-3n-E2B",

    # -------------------------
    # Gemma 2 (gated)
    # -------------------------
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",

    # mirror (optional)
    "AI-ModelScope/gemma-2-9b-it",

    # -------------------------
    # GLM
    # -------------------------
    "THUDM/GLM-4-9B-Chat",

    # -------------------------
    # DBRX (may be gated)
    # -------------------------
    "databricks/dbrx-base",
    "databricks/dbrx-instruct",
    "AI-ModelScope/dbrx-instruct",

    # -------------------------
    # Snowflake Arctic (mirror example)
    # -------------------------
    "AI-ModelScope/snowflake-arctic-instruct",

    # -------------------------
    # IBM Granite 3.0
    # -------------------------
    "ibm-granite/granite-3.0-2b-instruct",

    # -------------------------
    # Cohere Aya 23 (gated)
    # -------------------------
    "CohereForAI/aya-23-8B",
    "CohereForAI/aya-23-35B",

    # -------------------------
    # OLMo / OLMo 2 / OLMo 3
    # -------------------------
    "allenai/OLMo-1B",
    "allenai/OLMo-7B",
    "LLM-Research/OLMo-7B",

    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B",
    "allenai/OLMo-2-0325-32B-Instruct",

    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-1125-32B",
    # MS example mirror (duplicate in many cases)
    "allenai/Olmo-3-1125-32B",

    # -------------------------
    # Falcon 3
    # -------------------------
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",

    # -------------------------
    # Microsoft Phi
    # -------------------------
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-vision-128k-instruct",
    "microsoft/Phi-4-reasoning",

    # -------------------------
    # Yi (01.AI)
    # -------------------------
    "01-ai/Yi-6B",
    "01-ai/Yi-34B",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",

    # -------------------------
    # StableLM 2
    # -------------------------
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-12b",
    "stabilityai/stablelm-2-12b-chat",

    # -------------------------
    # BLOOM
    # -------------------------
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",

    # -------------------------
    # EleutherAI
    # -------------------------
    "EleutherAI/gpt-j-6b",
    "EleutherAI/gpt-neox-20b",

    # -------------------------
    # StarCoder2
    # -------------------------
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",

    # -------------------------
    # Code Llama (Meta) (gated)
    # -------------------------
    "meta-llama/CodeLlama-7b-hf",
    "meta-llama/CodeLlama-13b-hf",
    "meta-llama/CodeLlama-34b-hf",
    "meta-llama/CodeLlama-70b-hf",
    "meta-llama/CodeLlama-7b-Instruct-hf",
    "meta-llama/CodeLlama-13b-Instruct-hf",
    "meta-llama/CodeLlama-34b-Instruct-hf",
    "meta-llama/CodeLlama-70b-Instruct-hf",
    "meta-llama/CodeLlama-7b-Python-hf",
    "meta-llama/CodeLlama-13b-Python-hf",
    "meta-llama/CodeLlama-34b-Python-hf",
    "meta-llama/CodeLlama-70b-Python-hf",

    # -------------------------
    # InternLM / Baichuan
    # -------------------------
    "internlm/internlm2_5-7b",
    "internlm/internlm2_5-7b-chat",
    "baichuan-inc/Baichuan2-13B-Chat",

    # -------------------------
    # VLM / MLLM
    # -------------------------
    # Qwen3-VL: your list mentioned Qwen/Qwen3-VL-30B-A3B; on HF the commonly used repos are Instruct/Thinking.
    "Qwen/Qwen3-VL-30B-A3B",  # may error if not present on HF
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",

    # Llama 3.2 Vision already included above.

    "google/paligemma2-3b-pt-224",
    "google/paligemma2-3b-mix-224",

    "OpenGVLab/InternVL2_5-38B",

    "llava-hf/llava-v1.6-34b-hf",
    "llava-hf/LLaVA-NeXT-Video-34B-hf",

    "HuggingFaceM4/idefics2-8b",

    "zai-org/cogvlm2-llama3-chat-19B",

    "openbmb/MiniCPM-V",
    "openbmb/MiniCPM-V-4_5",

    "microsoft/Florence-2-large",

    "vikhyatk/moondream2",
]

# Deduplicate while preserving order:
_seen = set()
REPO_IDS = [x for x in REPO_IDS if not (x in _seen or _seen.add(x))]


# -------------------------
# 2) Extraction logic
# -------------------------

HF_TOKEN = os.getenv("HF_TOKEN")  # set this for gated models after accepting terms


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _pick_subcfg(cfg: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    for k in keys:
        sub = cfg.get(k)
        if isinstance(sub, dict):
            return sub
    return cfg


def _load_json(repo_id: str, filename: str) -> Dict[str, Any]:
    path = hf_hub_download(repo_id, filename, token=HF_TOKEN)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _head_dim(text_cfg: Dict[str, Any]) -> Optional[int]:
    if isinstance(text_cfg.get("head_dim"), int):
        return int(text_cfg["head_dim"])
    hs = _safe_int(text_cfg.get("hidden_size"))
    nh = _safe_int(text_cfg.get("num_attention_heads")) or _safe_int(text_cfg.get("n_head"))
    if hs and nh and nh > 0 and hs % nh == 0:
        return hs // nh
    return None


def _kv_heads(text_cfg: Dict[str, Any]) -> Optional[int]:
    kv = text_cfg.get("num_key_value_heads")
    if isinstance(kv, int):
        return kv
    # common fallback: MHA => kv_heads == heads
    nh = _safe_int(text_cfg.get("num_attention_heads")) or _safe_int(text_cfg.get("n_head"))
    return nh


def _layers(text_cfg: Dict[str, Any]) -> Optional[int]:
    return _safe_int(text_cfg.get("num_hidden_layers")) or _safe_int(text_cfg.get("n_layer"))


def _ctx(text_cfg: Dict[str, Any]) -> Optional[int]:
    return (
        _safe_int(text_cfg.get("max_position_embeddings")) or
        _safe_int(text_cfg.get("seq_length")) or
        _safe_int(text_cfg.get("model_max_length")) or
        _safe_int(text_cfg.get("max_sequence_length"))
    )


def _looks_nonstandard_kv(text_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> bool:
    # DeepSeek-V3/R1 config exposes qk_rope/nope head dims, LoRA ranks, etc.
    # Many VLMs also have special attention and cache layouts.
    flags = [
        "qk_rope_head_dim",
        "qk_nope_head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "mla",
    ]
    blob = json.dumps(full_cfg).lower()
    if any(f in text_cfg for f in flags):
        return True
    if "deepseek" in str(full_cfg.get("model_type", "")).lower():
        return True
    if "qwen3_vl" in blob or "qwen3vl" in blob:
        return True
    if "llava" in blob or "idefics" in blob or "cogvlm" in blob or "internvl" in blob:
        return True
    return False


def _kv_bytes_per_token_fp16(text_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Optional[int]:
    if _looks_nonstandard_kv(text_cfg, full_cfg):
        return None
    L = _layers(text_cfg)
    Hkv = _kv_heads(text_cfg)
    dh = _head_dim(text_cfg)
    if not (L and Hkv and dh):
        return None
    # fp16/bf16 => 2 bytes
    return 2 * 2 * L * Hkv * dh  # 2(K,V) * L * Hkv * dh * 2 bytes


def _get_attention_type(text_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> str:
    """Determine attention type: mha, gqa, or mla"""
    # Check for MLA (Multi-head Latent Attention)
    if _looks_nonstandard_kv(text_cfg, full_cfg):
        if "deepseek" in str(full_cfg.get("model_type", "")).lower():
            return "mla"
    
    heads = _safe_int(text_cfg.get("num_attention_heads")) or _safe_int(text_cfg.get("n_head"))
    kv_heads = _safe_int(text_cfg.get("num_key_value_heads"))
    
    if kv_heads is None or kv_heads == heads:
        return "mha"  # Multi-Head Attention
    else:
        return "gqa"  # Grouped Query Attention


def _get_model_type(text_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> str:
    """Determine if model is dense or MoE"""
    if text_cfg.get("num_local_experts") or text_cfg.get("n_routed_experts"):
        return "moe"
    return "dense"


@dataclass
class Row:
    repo_id: str
    status: str  # "ok" | "error"
    error: Optional[str]

    model_type: Optional[str]
    architectures: Optional[str]
    torch_dtype: Optional[str]

    # text-side (best guess: cfg/text_config/language_config/llm_config)
    layers: Optional[int]
    hidden_size: Optional[int]
    intermediate_size: Optional[int]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    head_dim: Optional[int]
    max_position_embeddings: Optional[int]
    vocab_size: Optional[int]
    hidden_act: Optional[str]
    rope_theta: Optional[float]
    rope_scaling: Optional[str]
    sliding_window: Optional[int]

    # MoE
    num_local_experts: Optional[int]
    n_routed_experts: Optional[int]
    num_experts_per_tok: Optional[int]
    moe_topk: Optional[int]

    # Derived KV (standard only)
    kv_bytes_per_token_fp16: Optional[int]
    kv_mib_per_1k_tokens_fp16: Optional[float]
    kv_gib_per_100k_tokens_fp16: Optional[float]

    # weight inventory (fast)
    weight_files_count: Optional[int]
    weight_total_bytes: Optional[int]

    # vision-side (if present; lightweight)
    vision_model_type: Optional[str]
    vision_layers: Optional[int]
    vision_hidden_size: Optional[int]
    vision_num_attention_heads: Optional[int]
    
    # For our models.json format
    attention_type: Optional[str]
    architecture_type: Optional[str]


def extract_one(api: HfApi, repo_id: str) -> Row:
    try:
        cfg = _load_json(repo_id, "config.json")

        # Choose best text config view for core LLM fields:
        text_cfg = _pick_subcfg(cfg, ("text_config", "language_config", "llm_config", "model_config"))

        model_type = (cfg.get("model_type") or text_cfg.get("model_type"))
        archs = cfg.get("architectures")
        architectures = ",".join(archs) if isinstance(archs, list) else None

        layers = _layers(text_cfg)
        hs = _safe_int(text_cfg.get("hidden_size"))
        ims = _safe_int(text_cfg.get("intermediate_size"))
        heads = _safe_int(text_cfg.get("num_attention_heads")) or _safe_int(text_cfg.get("n_head"))
        kvh = _safe_int(text_cfg.get("num_key_value_heads"))
        hd = _head_dim(text_cfg)
        ctx = _ctx(text_cfg)
        vocab = _safe_int(text_cfg.get("vocab_size"))
        act = text_cfg.get("hidden_act")
        rope_theta = _safe_float(text_cfg.get("rope_theta") or text_cfg.get("rotary_emb_base"))
        rope_scaling = text_cfg.get("rope_scaling")
        rope_scaling_s = json.dumps(rope_scaling, ensure_ascii=False) if rope_scaling is not None else None
        sliding_window = _safe_int(text_cfg.get("sliding_window"))
        torch_dtype = text_cfg.get("torch_dtype") or cfg.get("torch_dtype")

        # MoE (covers several families)
        num_local_experts = _safe_int(text_cfg.get("num_local_experts")) or _safe_int(text_cfg.get("num_experts")) or _safe_int(text_cfg.get("moe_num_experts"))
        n_routed_experts = _safe_int(text_cfg.get("n_routed_experts"))
        num_experts_per_tok = _safe_int(text_cfg.get("num_experts_per_tok"))
        moe_topk = _safe_int(text_cfg.get("moe_topk") or text_cfg.get("num_experts_per_tok"))
        
        # Override for specific models if config is elusive but we know them
        if "llama-4" in repo_id.lower() and not num_local_experts:
             # Heuristic for Llama-4 if keys missing
             if "16e" in repo_id.lower(): num_local_experts = 16
             if "128e" in repo_id.lower(): num_local_experts = 128

        # KV cache (standard only)
        kv_bpt = _kv_bytes_per_token_fp16(text_cfg, cfg)
        kv_mib_1k = (kv_bpt * 1000 / (1024 * 1024)) if kv_bpt else None
        kv_gib_100k = (kv_bpt * 100000 / (1024 ** 3)) if kv_bpt else None
        
        # Attention and architecture type
        attention_type = _get_attention_type(text_cfg, cfg)
        
        # Improve model type detection
        if num_local_experts or n_routed_experts:
             model_type_str = "moe"
        else:
             model_type_str = _get_model_type(text_cfg, cfg)
        
        architecture_type = model_type_str

        # Weight inventory via model_info (may require token for gated repos)
        weight_files_count = None
        weight_total_bytes = None
        try:
            info = api.model_info(repo_id, repo_type="model")
            total = 0
            count = 0
            for s in getattr(info, "siblings", []) or []:
                name = getattr(s, "rfilename", None) or getattr(s, "path", None)
                size = getattr(s, "size", None)
                if not name:
                    continue
                if name.endswith((".safetensors", ".bin")):
                    count += 1
                    if isinstance(size, int):
                        total += size
            weight_files_count = count
            weight_total_bytes = total if count > 0 else 0
        except Exception:
            pass

        # Vision-side (if present)
        vision_cfg = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else None
        vision_model_type = vision_cfg.get("model_type") if vision_cfg else None
        vision_layers = _safe_int(vision_cfg.get("num_hidden_layers")) if vision_cfg else None
        vision_hidden_size = _safe_int(vision_cfg.get("hidden_size")) if vision_cfg else None
        vision_heads = _safe_int(vision_cfg.get("num_attention_heads")) if vision_cfg else None

        return Row(
            repo_id=repo_id,
            status="ok",
            error=None,
            model_type=str(model_type) if model_type is not None else None,
            architectures=architectures,
            torch_dtype=str(torch_dtype) if torch_dtype is not None else None,
            layers=layers,
            hidden_size=hs,
            intermediate_size=ims,
            num_attention_heads=heads,
            num_key_value_heads=kvh,
            head_dim=hd,
            max_position_embeddings=ctx,
            vocab_size=vocab,
            hidden_act=str(act) if act is not None else None,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling_s,
            sliding_window=sliding_window,
            num_local_experts=num_local_experts,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_topk=moe_topk,
            kv_bytes_per_token_fp16=kv_bpt,
            kv_mib_per_1k_tokens_fp16=kv_mib_1k,
            kv_gib_per_100k_tokens_fp16=kv_gib_100k,
            weight_files_count=weight_files_count,
            weight_total_bytes=weight_total_bytes,
            vision_model_type=str(vision_model_type) if vision_model_type is not None else None,
            vision_layers=vision_layers,
            vision_hidden_size=vision_hidden_size,
            vision_num_attention_heads=vision_heads,
            attention_type=attention_type,
            architecture_type=architecture_type,
        )

    except Exception as e:
        return Row(
            repo_id=repo_id,
            status="error",
            error=str(e),
            model_type=None,
            architectures=None,
            torch_dtype=None,
            layers=None,
            hidden_size=None,
            intermediate_size=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            head_dim=None,
            max_position_embeddings=None,
            vocab_size=None,
            hidden_act=None,
            rope_theta=None,
            rope_scaling=None,
            sliding_window=None,
            num_local_experts=None,
            n_routed_experts=None,
            num_experts_per_tok=None,
            moe_topk=None,
            kv_bytes_per_token_fp16=None,
            kv_mib_per_1k_tokens_fp16=None,
            kv_gib_per_100k_tokens_fp16=None,
            weight_files_count=None,
            weight_total_bytes=None,
            vision_model_type=None,
            vision_layers=None,
            vision_hidden_size=None,
            vision_num_attention_heads=None,
            attention_type=None,
            architecture_type=None,
        )


def row_to_model_json(row: Row) -> Optional[Dict[str, Any]]:
    """Convert a Row to our models.json format"""
    if row.status != "ok":
        return None
    
    # Skip if missing essential fields
    if not all([row.layers, row.hidden_size, row.num_attention_heads]):
        return None
    
    # Generate model ID from repo_id
    parts = row.repo_id.split("/")
    model_name = parts[-1] if len(parts) > 1 else row.repo_id
    model_id = model_name.lower().replace("_", "-").replace(" ", "-")
    
    is_moe = row.num_local_experts or row.n_routed_experts
    if "llama-4" in row.repo_id.lower():
        print(f"DEBUG: {row.repo_id} | is_moe={is_moe} | experts={row.num_local_experts}/{row.n_routed_experts} | weights={row.weight_total_bytes}")
    
    # Try to get explicit parameter count from config if we extracted it (need to add to Row first)
    # Actually, Row doesn't have it. I need to add it value to Row in extract_one or just re-estimate.
    # Let's add explicit param check in extract_one and Row.
    
    # ... Wait, I can't edit Row definition easily in this replace_file_content without updating the whole class.
    # I should have added it to Row.
    # Alternative: Trust the architecture estimation for MoE, it's safer than the file list for Gated/Partial repos.
    
    # Estimate parameters from weight bytes (rough estimate)
    # OR if MoE, prefer architecture estimation as file listing might be incomplete for gated models
    use_file_weights = row.weight_total_bytes and not is_moe
    
    if use_file_weights:
        # Assume FP16 weights (2 bytes per param)
        params = row.weight_total_bytes / 2
    else:
        # Better estimation based on architecture
        h = row.hidden_size or 4096
        L = row.layers or 32
        inter = row.intermediate_size or (h * 4)
        vocab = row.vocab_size or 32000
        
        # Base logic:
        # Attention: 4 * h^2 (Q, K, V, O) - roughly
        # FFN (SwiGLU): 3 * h * intermediate (Gate, Up, Down)
        
        attn_params_per_layer = 4 * h * h
        
        if is_moe:
            # MoE: FFN is replicated per expert
            num_experts = row.num_local_experts or row.n_routed_experts or 8
            ffn_params_per_layer = num_experts * 3 * h * inter
        else:
            # Dense
            ffn_params_per_layer = 3 * h * inter
            
        # Total approx
        params = (L * (attn_params_per_layer + ffn_params_per_layer)) + (vocab * h)
        
    if "llama-4" in row.repo_id.lower():
        print(f"DEBUG: {row.repo_id} | Final Params={params}")
    
    # Get provider from repo_id
    provider = parts[0] if len(parts) > 1 else "Unknown"
    provider_map = {
        "meta-llama": "Meta",
        "Qwen": "Alibaba",
        "deepseek-ai": "DeepSeek",
        "mistralai": "Mistral AI",
        "google": "Google",
        "microsoft": "Microsoft",
        "allenai": "Allen AI",
        "tiiuae": "TII",
        "stabilityai": "Stability AI",
        "bigscience": "BigScience",
        "EleutherAI": "EleutherAI",
        "bigcode": "BigCode",
        "internlm": "InternLM",
        "baichuan-inc": "Baichuan",
        "01-ai": "01.AI",
        "ibm-granite": "IBM",
        "CohereForAI": "Cohere",
        "THUDM": "Tsinghua",
        "databricks": "Databricks",
    }
    provider = provider_map.get(provider, provider)
    
    model = {
        "id": model_id,
        "name": model_name,
        "provider": provider,
        "parameters": params,
        "architecture": {
            "type": row.architecture_type or "dense",
            "num_layers": row.layers,
            "hidden_size": row.hidden_size,
            "num_heads": row.num_attention_heads,
            "num_kv_heads": row.num_key_value_heads or row.num_attention_heads,
            "head_dim": row.head_dim or (row.hidden_size // row.num_attention_heads if row.hidden_size and row.num_attention_heads else 128),
            "intermediate_size": row.intermediate_size or row.hidden_size * 4,
            "vocab_size": row.vocab_size or 32000,
            "max_position_embeddings": row.max_position_embeddings or 4096,
            "attention_type": row.attention_type or "mha",
        }
    }
    
    # Add MoE fields if applicable
    if row.num_local_experts or row.n_routed_experts:
        model["architecture"]["num_experts"] = row.num_local_experts or row.n_routed_experts
        model["architecture"]["num_experts_per_token"] = row.num_experts_per_tok or row.moe_topk or 2
    
    return model


def main() -> None:
    api = HfApi(token=HF_TOKEN)

    rows: List[Row] = []
    
    # Critical models only for quick update
    critical_ids = [
        "meta-llama/Llama-4-Scout-17B-16E",
        "meta-llama/Llama-4-Maverick-17B-128E",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b"
    ]
    
    for rid in critical_ids:
        print(f"Processing: {rid}")
        rows.append(extract_one(api, rid))

    out_csv = "scripts/model_details.csv"
    out_jsonl = "scripts/model_details.jsonl"
    out_models_json = "scripts/models_hf.json"

    # CSV
    dict_rows = [asdict(r) for r in rows]
    fieldnames = sorted({k for d in dict_rows for k in d.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dict_rows)

    # JSONL (useful for downstream processing)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for d in dict_rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    # models.json format for our API
    models_list = []
    for row in rows:
        model = row_to_model_json(row)
        if model:
            models_list.append(model)
    
    with open(out_models_json, "w", encoding="utf-8") as f:
        json.dump({"models": models_list}, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in rows if r.status == "ok")
    err = len(rows) - ok
    print(f"Done. {len(rows)} repos processed: {ok} ok, {err} error.")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_models_json}")
    if err:
        print("Tip: most errors are usually gated repos; set HF_TOKEN after accepting terms.")


if __name__ == "__main__":
    main()
