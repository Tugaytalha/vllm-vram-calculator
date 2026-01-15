"""
vLLM VRAM Calculator API
"""

from .main import app
from .calculator import (
    calculate_vram,
    calculate_max_concurrent_users,
    CalculationConfig,
    ModelArchitecture,
    VRAMBreakdown,
    QuantizationType,
    KVCacheQuantization,
)

__version__ = "0.1.0"
__all__ = [
    "app",
    "calculate_vram",
    "calculate_max_concurrent_users",
    "CalculationConfig",
    "ModelArchitecture",
    "VRAMBreakdown",
    "QuantizationType",
    "KVCacheQuantization",
]
