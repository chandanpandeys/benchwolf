"""Model database for hardware feasibility checks.

Contains parameter counts and RAM estimates for popular Ollama models.
No model download required — sizes are computed from public specs.

RAM formula:
    model_ram_gb ≈ (params_B × bits_per_weight / 8) + overhead_gb
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    """Specification for a known model."""
    name: str                 # Ollama model name (e.g., "qwen2.5:3b")
    params_b: float           # Parameter count in billions
    default_quant: str        # Default quantization (e.g., "Q4_K_M")
    family: str               # Model family (e.g., "Qwen 2.5")
    description: str          # Short description
    min_ram_gb: float         # Minimum RAM needed (with quant + overhead)
    recommended_ram_gb: float # Recommended RAM (with headroom)
    disk_size_gb: float       # Approximate download size

    @property
    def status_for_ram(self) -> str:
        """Get a human-readable label for this model."""
        return f"{self.name} ({self.default_quant}, {self.params_b}B)"


def _estimate_ram(params_b: float, bits: int = 4, overhead_gb: float = 0.5) -> float:
    """Estimate RAM needed for a model.

    Args:
        params_b: Parameters in billions.
        bits: Bits per weight (4 for Q4, 8 for Q8, 16 for F16).
        overhead_gb: OS/runtime overhead.

    Returns:
        Estimated RAM in GB.
    """
    model_gb = (params_b * bits) / 8
    return round(model_gb + overhead_gb, 1)


# ── Model Database ────────────────────────────────────────────────────
# Sorted by RAM requirement (ascending), covering the most popular
# Ollama models that Indian MSME users are likely to encounter.

MODEL_DATABASE: list[ModelSpec] = [
    ModelSpec(
        name="qwen2.5:0.5b",
        params_b=0.5,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Tiny model, great for testing",
        min_ram_gb=_estimate_ram(0.5),
        recommended_ram_gb=2.0,
        disk_size_gb=0.4,
    ),
    ModelSpec(
        name="qwen2.5:1.5b",
        params_b=1.5,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Small but capable",
        min_ram_gb=_estimate_ram(1.5),
        recommended_ram_gb=3.0,
        disk_size_gb=1.0,
    ),
    ModelSpec(
        name="gemma2:2b",
        params_b=2.0,
        default_quant="Q4_K_M",
        family="Gemma 2",
        description="Google's efficient small model",
        min_ram_gb=_estimate_ram(2.0),
        recommended_ram_gb=3.0,
        disk_size_gb=1.6,
    ),
    ModelSpec(
        name="phi3:mini",
        params_b=3.8,
        default_quant="Q4_K_M",
        family="Phi-3",
        description="Microsoft's excellent small model",
        min_ram_gb=_estimate_ram(3.8),
        recommended_ram_gb=4.0,
        disk_size_gb=2.3,
    ),
    ModelSpec(
        name="qwen2.5:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Best quality-to-size ratio",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="llama3.2:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Llama 3.2",
        description="Meta's latest small model",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="qwen2.5-coder:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5 Coder",
        description="Code-specialized model",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="deepseek-r1:1.5b",
        params_b=1.5,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="Reasoning-focused model",
        min_ram_gb=_estimate_ram(1.5),
        recommended_ram_gb=3.0,
        disk_size_gb=1.1,
    ),
    ModelSpec(
        name="mistral:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="Mistral",
        description="Strong general-purpose model",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.1,
    ),
    ModelSpec(
        name="llama3.1:8b",
        params_b=8.0,
        default_quant="Q4_K_M",
        family="Llama 3.1",
        description="Meta's flagship open model",
        min_ram_gb=_estimate_ram(8.0),
        recommended_ram_gb=8.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="gemma2:9b",
        params_b=9.0,
        default_quant="Q4_K_M",
        family="Gemma 2",
        description="Google's mid-size model",
        min_ram_gb=_estimate_ram(9.0),
        recommended_ram_gb=8.0,
        disk_size_gb=5.4,
    ),
    ModelSpec(
        name="qwen2.5:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Excellent multilingual model",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="deepseek-r1:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="Reasoning at mid-size",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="deepseek-r1:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="Larger reasoning model",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.0,
    ),
    ModelSpec(
        name="qwen2.5:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="High-quality mid-size",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.0,
    ),
    ModelSpec(
        name="llama3.1:70b",
        params_b=70.0,
        default_quant="Q4_K_M",
        family="Llama 3.1",
        description="Flagship large model (needs beefy hardware)",
        min_ram_gb=_estimate_ram(70.0),
        recommended_ram_gb=48.0,
        disk_size_gb=40.0,
    ),
]


def get_model_spec(name: str) -> Optional[ModelSpec]:
    """Look up a model by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    for spec in MODEL_DATABASE:
        if spec.name.lower() == name_lower:
            return spec
    # Partial match
    for spec in MODEL_DATABASE:
        if name_lower in spec.name.lower() or spec.name.lower() in name_lower:
            return spec
    return None


def estimate_custom_model_ram(params_b: float, quant_bits: int = 4) -> float:
    """Estimate RAM for a model not in the database."""
    return _estimate_ram(params_b, bits=quant_bits)
