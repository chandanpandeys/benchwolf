"""Model database for hardware feasibility checks.

Contains parameter counts, MoE characteristics, and RAM estimates for popular Ollama models.
No model download required — sizes are computed from public architecture specs.

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
    params_b: float           # Total parameter count in billions
    default_quant: str        # Default quantization (e.g., "Q4_K_M")
    family: str               # Model family (e.g., "Qwen 2.5")
    description: str          # Short description
    min_ram_gb: float         # Minimum RAM needed (with quant + overhead)
    recommended_ram_gb: float # Recommended RAM (with headroom)
    disk_size_gb: float       # Approximate download size
    active_params_b: Optional[float] = None  # Active params for MoE models
    is_moe: bool = False      # True if Mixture-of-Experts

    @property
    def effective_inference_params_b(self) -> float:
        """Effective parameter count processed per token."""
        return self.active_params_b if self.active_params_b is not None else self.params_b

    @property
    def status_for_ram(self) -> str:
        """Get a human-readable label for this model."""
        moe_str = f" [MoE {self.active_params_b}B active]" if self.is_moe and self.active_params_b else ""
        return f"{self.name} ({self.default_quant}, {self.params_b}B{moe_str})"


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
# models across edge, laptop, and workstation hardware.

MODEL_DATABASE: list[ModelSpec] = [
    # ── Ultra-Lightweight (Under 1B) ──
    ModelSpec(
        name="smollm2:135m",
        params_b=0.135,
        default_quant="Q4_K_M",
        family="SmolLM2",
        description="HuggingFace SmolLM2 ultra-compact model",
        min_ram_gb=_estimate_ram(0.135, overhead_gb=0.2),
        recommended_ram_gb=1.0,
        disk_size_gb=0.1,
    ),
    ModelSpec(
        name="smollm2:360m",
        params_b=0.36,
        default_quant="Q4_K_M",
        family="SmolLM2",
        description="Compact on-device model",
        min_ram_gb=_estimate_ram(0.36, overhead_gb=0.3),
        recommended_ram_gb=1.5,
        disk_size_gb=0.25,
    ),
    ModelSpec(
        name="qwen2.5:0.5b",
        params_b=0.5,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Tiny model, great for edge and microcontrollers",
        min_ram_gb=_estimate_ram(0.5),
        recommended_ram_gb=2.0,
        disk_size_gb=0.4,
    ),
    ModelSpec(
        name="qwen3:0.6b",
        params_b=0.6,
        default_quant="Q4_K_M",
        family="Qwen 3",
        description="Next-gen Qwen tiny model",
        min_ram_gb=_estimate_ram(0.6),
        recommended_ram_gb=2.0,
        disk_size_gb=0.5,
    ),

    # ── Small On-Device (1B - 2B) ──
    ModelSpec(
        name="llama3.2:1b",
        params_b=1.0,
        default_quant="Q4_K_M",
        family="Llama 3.2",
        description="Meta's ultra-lightweight 1B model",
        min_ram_gb=_estimate_ram(1.0),
        recommended_ram_gb=2.5,
        disk_size_gb=0.8,
    ),
    ModelSpec(
        name="gemma3:1b",
        params_b=1.0,
        default_quant="Q4_K_M",
        family="Gemma 3",
        description="Google Gemma 3 multimodal compact model",
        min_ram_gb=_estimate_ram(1.0),
        recommended_ram_gb=2.5,
        disk_size_gb=0.8,
    ),
    ModelSpec(
        name="smollm2:1.7b",
        params_b=1.7,
        default_quant="Q4_K_M",
        family="SmolLM2",
        description="High-reasoning small SLM",
        min_ram_gb=_estimate_ram(1.7),
        recommended_ram_gb=3.0,
        disk_size_gb=1.0,
    ),
    ModelSpec(
        name="qwen2.5:1.5b",
        params_b=1.5,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Small but capable general model",
        min_ram_gb=_estimate_ram(1.5),
        recommended_ram_gb=3.0,
        disk_size_gb=1.0,
    ),
    ModelSpec(
        name="deepseek-r1:1.5b",
        params_b=1.5,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="Reasoning & thinking SLM model",
        min_ram_gb=_estimate_ram(1.5),
        recommended_ram_gb=3.0,
        disk_size_gb=1.1,
    ),
    ModelSpec(
        name="qwen3:1.8b",
        params_b=1.8,
        default_quant="Q4_K_M",
        family="Qwen 3",
        description="Qwen 3 reasoning and instruction model",
        min_ram_gb=_estimate_ram(1.8),
        recommended_ram_gb=3.0,
        disk_size_gb=1.2,
    ),
    ModelSpec(
        name="gemma2:2b",
        params_b=2.0,
        default_quant="Q4_K_M",
        family="Gemma 2",
        description="Google's efficient 2B small model",
        min_ram_gb=_estimate_ram(2.0),
        recommended_ram_gb=3.0,
        disk_size_gb=1.6,
    ),

    # ── Mid-Low Tier (3B - 4B) ──
    ModelSpec(
        name="qwen2.5:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Best quality-to-size ratio on laptops",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="llama3.2:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Llama 3.2",
        description="Meta's popular 3B edge model",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="qwen2.5-coder:3b",
        params_b=3.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5 Coder",
        description="Code-specialized 3B model",
        min_ram_gb=_estimate_ram(3.0),
        recommended_ram_gb=4.0,
        disk_size_gb=2.0,
    ),
    ModelSpec(
        name="phi3:mini",
        params_b=3.8,
        default_quant="Q4_K_M",
        family="Phi-3",
        description="Microsoft's compact powerhouse",
        min_ram_gb=_estimate_ram(3.8),
        recommended_ram_gb=4.0,
        disk_size_gb=2.3,
    ),
    ModelSpec(
        name="phi3.5:3.8b",
        params_b=3.8,
        default_quant="Q4_K_M",
        family="Phi-3.5",
        description="Microsoft Phi-3.5 with 128k context",
        min_ram_gb=_estimate_ram(3.8),
        recommended_ram_gb=4.5,
        disk_size_gb=2.3,
    ),
    ModelSpec(
        name="phi4-mini:3.8b",
        params_b=3.8,
        default_quant="Q4_K_M",
        family="Phi-4",
        description="Microsoft Phi-4 mini high-reasoning model",
        min_ram_gb=_estimate_ram(3.8),
        recommended_ram_gb=4.5,
        disk_size_gb=2.4,
    ),
    ModelSpec(
        name="gemma3:4b",
        params_b=4.0,
        default_quant="Q4_K_M",
        family="Gemma 3",
        description="Google Gemma 3 vision-language model",
        min_ram_gb=_estimate_ram(4.0),
        recommended_ram_gb=5.0,
        disk_size_gb=2.6,
    ),
    ModelSpec(
        name="qwen3:4b",
        params_b=4.0,
        default_quant="Q4_K_M",
        family="Qwen 3",
        description="Qwen 3 versatile desktop model",
        min_ram_gb=_estimate_ram(4.0),
        recommended_ram_gb=5.0,
        disk_size_gb=2.5,
    ),

    # ── Standard Mainstream (7B - 9B) ──
    ModelSpec(
        name="mistral:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="Mistral",
        description="Strong general-purpose classic model",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.1,
    ),
    ModelSpec(
        name="qwen2.5:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="Exceptional 7B multilingual and reasoning model",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="qwen2.5-coder:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5 Coder",
        description="Leader in 7B coding benchmarks",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="deepseek-r1:7b",
        params_b=7.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="DeepSeek R1 distilled on Qwen 7B",
        min_ram_gb=_estimate_ram(7.0),
        recommended_ram_gb=6.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="llama3.1:8b",
        params_b=8.0,
        default_quant="Q4_K_M",
        family="Llama 3.1",
        description="Meta's flagship open-weights 8B model",
        min_ram_gb=_estimate_ram(8.0),
        recommended_ram_gb=8.0,
        disk_size_gb=4.7,
    ),
    ModelSpec(
        name="deepseek-r1:8b",
        params_b=8.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="DeepSeek R1 reasoning distilled on Llama 3.1 8B",
        min_ram_gb=_estimate_ram(8.0),
        recommended_ram_gb=8.0,
        disk_size_gb=4.9,
    ),
    ModelSpec(
        name="qwen3:8b",
        params_b=8.0,
        default_quant="Q4_K_M",
        family="Qwen 3",
        description="Qwen 3 standard workhorse model",
        min_ram_gb=_estimate_ram(8.0),
        recommended_ram_gb=8.0,
        disk_size_gb=5.0,
    ),
    ModelSpec(
        name="gemma2:9b",
        params_b=9.0,
        default_quant="Q4_K_M",
        family="Gemma 2",
        description="Google's high-capability 9B model",
        min_ram_gb=_estimate_ram(9.0),
        recommended_ram_gb=8.0,
        disk_size_gb=5.4,
    ),

    # ── Heavyweights & MoE (12B - 16B) ──
    ModelSpec(
        name="gemma3:12b",
        params_b=12.0,
        default_quant="Q4_K_M",
        family="Gemma 3",
        description="Google Gemma 3 vision-text 12B model",
        min_ram_gb=_estimate_ram(12.0),
        recommended_ram_gb=10.0,
        disk_size_gb=7.5,
    ),
    ModelSpec(
        name="mistral-nemo:12b",
        params_b=12.0,
        default_quant="Q4_K_M",
        family="Mistral",
        description="Mistral AI & NVIDIA 12B collaboration",
        min_ram_gb=_estimate_ram(12.0),
        recommended_ram_gb=10.0,
        disk_size_gb=7.1,
    ),
    ModelSpec(
        name="deepseek-r1:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="DeepSeek R1 reasoning model (14B)",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.0,
    ),
    ModelSpec(
        name="qwen2.5:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5",
        description="High-quality 14B instruction model",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.0,
    ),
    ModelSpec(
        name="qwen2.5-coder:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5 Coder",
        description="State-of-the-art 14B coding specialist",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.0,
    ),
    ModelSpec(
        name="phi4:14b",
        params_b=14.0,
        default_quant="Q4_K_M",
        family="Phi-4",
        description="Microsoft's premier 14B reasoning model",
        min_ram_gb=_estimate_ram(14.0),
        recommended_ram_gb=12.0,
        disk_size_gb=9.1,
    ),

    # ── High-End & Workstation Tier (27B - 70B+) ──
    ModelSpec(
        name="gemma3:27b",
        params_b=27.0,
        default_quant="Q4_K_M",
        family="Gemma 3",
        description="Google Gemma 3 flagship multimodal model",
        min_ram_gb=_estimate_ram(27.0),
        recommended_ram_gb=20.0,
        disk_size_gb=16.0,
    ),
    ModelSpec(
        name="deepseek-r1:32b",
        params_b=32.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="High-performance 32B reasoning model",
        min_ram_gb=_estimate_ram(32.0),
        recommended_ram_gb=24.0,
        disk_size_gb=20.0,
    ),
    ModelSpec(
        name="qwen2.5-coder:32b",
        params_b=32.0,
        default_quant="Q4_K_M",
        family="Qwen 2.5 Coder",
        description="Enterprise grade 32B coding model",
        min_ram_gb=_estimate_ram(32.0),
        recommended_ram_gb=24.0,
        disk_size_gb=20.0,
    ),
    ModelSpec(
        name="llama3.3:70b",
        params_b=70.0,
        default_quant="Q4_K_M",
        family="Llama 3.3",
        description="Meta's latest 70B flagship model",
        min_ram_gb=_estimate_ram(70.0),
        recommended_ram_gb=48.0,
        disk_size_gb=42.0,
    ),
    ModelSpec(
        name="llama4:scout",
        params_b=54.0,
        default_quant="Q4_K_M",
        family="Llama 4",
        description="Meta Llama 4 native multimodal MoE architecture",
        min_ram_gb=_estimate_ram(54.0),
        recommended_ram_gb=36.0,
        disk_size_gb=32.0,
        active_params_b=12.0,
        is_moe=True,
    ),
    ModelSpec(
        name="llama3.1:70b",
        params_b=70.0,
        default_quant="Q4_K_M",
        family="Llama 3.1",
        description="Flagship large open model (needs unified/multi-GPU)",
        min_ram_gb=_estimate_ram(70.0),
        recommended_ram_gb=48.0,
        disk_size_gb=40.0,
    ),
    ModelSpec(
        name="deepseek-r1:70b",
        params_b=70.0,
        default_quant="Q4_K_M",
        family="DeepSeek R1",
        description="DeepSeek R1 reasoning distilled on Llama 70B",
        min_ram_gb=_estimate_ram(70.0),
        recommended_ram_gb=48.0,
        disk_size_gb=43.0,
    ),
    ModelSpec(
        name="deepseek-v3:671b",
        params_b=671.0,
        default_quant="Q4_K_M",
        family="DeepSeek V3",
        description="DeepSeek V3 671B MoE (37B active parameters)",
        min_ram_gb=_estimate_ram(671.0),
        recommended_ram_gb=400.0,
        disk_size_gb=380.0,
        active_params_b=37.0,
        is_moe=True,
    ),
]


def get_model_spec(name: str) -> Optional[ModelSpec]:
    """Look up a model by name (case-insensitive, exact then partial match)."""
    name_lower = name.lower().strip()

    # Exact match first
    for spec in MODEL_DATABASE:
        if spec.name.lower() == name_lower:
            return spec

    # Prefix / exact tag match (e.g. "qwen2.5:3b" matches "qwen2.5:3b-instruct-q4_K_M")
    for spec in MODEL_DATABASE:
        if spec.name.lower().split(":")[0] == name_lower.split(":")[0]:
            if ":" in name_lower and ":" in spec.name:
                if spec.name.lower().split(":")[1] in name_lower.split(":")[1]:
                    return spec

    # Partial substring match
    for spec in MODEL_DATABASE:
        if name_lower in spec.name.lower() or spec.name.lower() in name_lower:
            return spec

    return None


def get_quant_ram_table(params_b: float) -> dict[str, float]:
    """Get RAM estimates across common quantization formats for a model."""
    return {
        "IQ3_M (3-bit)": _estimate_ram(params_b, bits=3, overhead_gb=0.4),
        "Q4_K_M (4-bit default)": _estimate_ram(params_b, bits=4, overhead_gb=0.5),
        "Q5_K_M (5-bit)": _estimate_ram(params_b, bits=5, overhead_gb=0.5),
        "Q8_0 / FP8 (8-bit)": _estimate_ram(params_b, bits=8, overhead_gb=0.6),
        "FP16 (16-bit unquantized)": _estimate_ram(params_b, bits=16, overhead_gb=0.8),
    }


def estimate_custom_model_ram(params_b: float, quant_bits: int = 4) -> float:
    """Estimate RAM for a model not in the database."""
    return _estimate_ram(params_b, bits=quant_bits)
