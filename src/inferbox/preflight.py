"""Preflight hardware feasibility checker.

Checks if the user's hardware can run specific LLM models
WITHOUT downloading anything — no Ollama, no model files.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from inferbox.data.model_db import MODEL_DATABASE, ModelSpec, get_model_spec
from inferbox.hardware.detect import detect_hardware
from inferbox.models import HardwareProfile


class FitStatus(Enum):
    """How well a model fits the hardware."""
    EASY = "easy"         # Plenty of headroom
    GOOD = "good"         # Fits with reasonable headroom
    TIGHT = "tight"       # Fits, but will be slow / might swap
    NO_FIT = "no_fit"     # Won't fit in RAM
    NO_DISK = "no_disk"   # Not enough disk space to download


@dataclass
class ModelFitResult:
    """Result of checking one model against the hardware."""
    model: ModelSpec
    status: FitStatus
    available_ram_gb: float
    ram_headroom_gb: float      # How much RAM is left after loading
    disk_ok: bool               # Enough disk space?
    estimated_tok_s: Optional[float] = None  # Expected tok/s estimate
    bandwidth_ceiling_tok_s: Optional[float] = None  # Theoretical peak tok/s
    reason: str = ""

    @property
    def status_emoji(self) -> str:
        return {
            FitStatus.EASY: "✅",
            FitStatus.GOOD: "✅",
            FitStatus.TIGHT: "⚠️ ",
            FitStatus.NO_FIT: "❌",
            FitStatus.NO_DISK: "💾",
        }[self.status]

    @property
    def status_label(self) -> str:
        return {
            FitStatus.EASY: "Easy — will run smoothly",
            FitStatus.GOOD: "Good — enough headroom",
            FitStatus.TIGHT: "Tight — may be slow or swap",
            FitStatus.NO_FIT: "Won't fit — not enough RAM",
            FitStatus.NO_DISK: "Not enough disk space",
        }[self.status]


@dataclass
class PreflightReport:
    """Full preflight check results."""
    hardware: HardwareProfile
    available_ram_gb: float      # What's free RIGHT NOW
    usable_ram_gb: float         # Realistic max (total - OS overhead)
    free_disk_gb: float
    model_results: list[ModelFitResult]
    recommended_model: Optional[str] = None
    can_run_any: bool = False
    ollama_running: bool = False


def _estimate_tok_s(model: ModelSpec, hw: HardwareProfile) -> tuple[float, Optional[float]]:
    """Estimate throughput (realistic tok/s and theoretical memory bandwidth ceiling).

    Text generation speed for LLMs is fundamentally memory-bandwidth bound:
        Theoretical Max Tok/s = Memory Bandwidth (GB/s) / Active Model Weights (GB)

    Args:
        model: Model specifications.
        hw: Detected hardware profile.

    Returns:
        (estimated_realistic_tok_s, theoretical_bandwidth_ceiling_tok_s)
    """
    effective_params = model.effective_inference_params_b
    weight_size_gb = max(effective_params * 0.55, 0.08)  # ~0.55 GB per 1B params at Q4_K_M

    bandwidth_ceiling = None
    if hw.memory_bandwidth_gb_s and hw.memory_bandwidth_gb_s > 0:
        bandwidth_ceiling = round(hw.memory_bandwidth_gb_s / weight_size_gb, 1)

    # 1. Hardware acceleration estimation (Apple Silicon / Dedicated GPU)
    if hw.gpu_name or "apple" in hw.cpu_name.lower() or "darwin" in hw.os_name.lower():
        # High-performance hardware typically reaches ~45-75% of theoretical bandwidth efficiency
        if bandwidth_ceiling:
            realistic = bandwidth_ceiling * 0.58
            return round(max(realistic, 1.0), 1), bandwidth_ceiling

    # 2. CPU / Standard RAM heuristic estimation
    base_tok_s = 6.0
    base_params = 3.0
    base_cores = 4

    size_factor = base_params / max(effective_params, 0.1)
    cores = max(hw.cpu_cores_physical, 1)
    core_factor = min(cores / base_cores, 2.5)

    freq_factor = 1.0
    if hw.cpu_freq_mhz and hw.cpu_freq_mhz > 0:
        freq_factor = min(hw.cpu_freq_mhz / 2500.0, 1.6)

    estimated = base_tok_s * size_factor * core_factor * freq_factor

    # Cap by bandwidth ceiling if known
    if bandwidth_ceiling and estimated > bandwidth_ceiling:
        estimated = bandwidth_ceiling * 0.5

    return round(max(estimated, 0.5), 1), bandwidth_ceiling


def _check_model_fit(
    model: ModelSpec,
    hw: HardwareProfile,
    usable_ram_gb: float,
    free_disk_gb: float,
) -> ModelFitResult:
    """Check if a single model fits on this hardware."""
    # Check disk space first
    if free_disk_gb < model.disk_size_gb + 1.0:  # Need model + 1GB buffer
        return ModelFitResult(
            model=model,
            status=FitStatus.NO_DISK,
            available_ram_gb=usable_ram_gb,
            ram_headroom_gb=usable_ram_gb - model.min_ram_gb,
            disk_ok=False,
            reason=f"Need {model.disk_size_gb:.1f} GB disk, only {free_disk_gb:.1f} GB free",
        )

    headroom = usable_ram_gb - model.min_ram_gb

    # Determine fit status
    if headroom < 0:
        status = FitStatus.NO_FIT
        reason = f"Need {model.min_ram_gb:.1f} GB, only ~{usable_ram_gb:.1f} GB usable"
    elif headroom < 1.0:
        status = FitStatus.TIGHT
        reason = f"Only {headroom:.1f} GB headroom"
    elif headroom < 2.5:
        status = FitStatus.GOOD
        reason = f"{headroom:.1f} GB headroom"
    else:
        status = FitStatus.EASY
        reason = f"{headroom:.1f} GB headroom"

    tok_s = None
    ceiling = None
    if status != FitStatus.NO_FIT:
        tok_s, ceiling = _estimate_tok_s(model, hw)

    return ModelFitResult(
        model=model,
        status=status,
        available_ram_gb=usable_ram_gb,
        ram_headroom_gb=round(headroom, 1),
        disk_ok=True,
        estimated_tok_s=tok_s,
        bandwidth_ceiling_tok_s=ceiling,
        reason=reason,
    )


def run_preflight(
    specific_model: Optional[str] = None,
) -> PreflightReport:
    """Run a full preflight hardware check.

    Args:
        specific_model: Optional model name to check specifically.
                       If None, checks all models in the database.

    Returns:
        PreflightReport with compatibility results.
    """
    hw = detect_hardware()

    # Current available RAM (may be low if apps are open)
    available_ram = hw.ram_available_gb

    # Usable RAM = total minus ~2 GB OS/system overhead.
    # This is a realistic estimate of what Ollama can use
    # when non-essential apps are closed.
    os_overhead_gb = 2.0
    usable_ram = max(hw.ram_total_gb - os_overhead_gb, 0.5)

    # Get free disk space
    free_disk = 0.0
    try:
        disk = shutil.disk_usage("/")
        free_disk = round(disk.free / (1024 ** 3), 1)
    except Exception:
        try:
            disk = shutil.disk_usage(os.path.expanduser("~"))
            free_disk = round(disk.free / (1024 ** 3), 1)
        except Exception:
            free_disk = 0.0

    # Check which models to evaluate
    if specific_model:
        spec = get_model_spec(specific_model)
        if spec:
            models_to_check = [spec]
        else:
            models_to_check = MODEL_DATABASE
    else:
        models_to_check = MODEL_DATABASE

    # Check models against usable RAM
    results: list[ModelFitResult] = []
    for model in models_to_check:
        result = _check_model_fit(model, hw, usable_ram, free_disk)
        results.append(result)

    # Find recommendation (best model that fits with GOOD or EASY)
    recommended = None
    runnable = [r for r in results if r.status in (FitStatus.EASY, FitStatus.GOOD)]
    if runnable:
        # Pick the largest model that still fits well — best quality
        best = max(runnable, key=lambda r: r.model.params_b)
        recommended = best.model.name

    can_run_any = any(r.status in (FitStatus.EASY, FitStatus.GOOD, FitStatus.TIGHT) for r in results)

    # Check if Ollama is running
    ollama_running = False
    try:
        from inferbox.backends.ollama import OllamaBackend
        ollama_running = OllamaBackend().is_available()
    except Exception:
        pass

    return PreflightReport(
        hardware=hw,
        available_ram_gb=available_ram,
        usable_ram_gb=round(usable_ram, 1),
        free_disk_gb=free_disk,
        model_results=results,
        recommended_model=recommended,
        can_run_any=can_run_any,
        ollama_running=ollama_running,
    )
