"""Preflight hardware feasibility checker.

Checks if the user's hardware can run specific LLM models
WITHOUT downloading anything — no Ollama, no model files.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tinybench.data.model_db import MODEL_DATABASE, ModelSpec, get_model_spec
from tinybench.hardware.detect import detect_hardware
from tinybench.models import HardwareProfile


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
    estimated_tok_s: Optional[float] = None  # Rough tok/s estimate
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


def _estimate_tok_s(model: ModelSpec, hw: HardwareProfile) -> float:
    """Very rough tok/s estimate based on hardware.

    Uses a simple heuristic:
    - Base: ~5 tok/s for a 3B Q4 model on a modern 4-core CPU
    - Scale linearly with cores, inversely with model size
    - Bonus for higher CPU frequency
    """
    base_tok_s = 5.0
    base_params = 3.0
    base_cores = 4

    # Scale by model size (inversely proportional)
    size_factor = base_params / max(model.params_b, 0.1)

    # Scale by available cores
    cores = max(hw.cpu_cores_physical, 1)
    core_factor = min(cores / base_cores, 2.0)  # Cap at 2x

    # Frequency bonus (baseline 2.5 GHz)
    freq_factor = 1.0
    if hw.cpu_freq_mhz and hw.cpu_freq_mhz > 0:
        freq_factor = min(hw.cpu_freq_mhz / 2500, 1.5)

    estimated = base_tok_s * size_factor * core_factor * freq_factor
    return round(max(estimated, 0.5), 1)


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

    tok_s = _estimate_tok_s(model, hw) if status != FitStatus.NO_FIT else None

    return ModelFitResult(
        model=model,
        status=status,
        available_ram_gb=usable_ram_gb,
        ram_headroom_gb=round(headroom, 1),
        disk_ok=True,
        estimated_tok_s=tok_s,
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
            # Windows fallback
            import os
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
            # Unknown model — check all
            models_to_check = MODEL_DATABASE
    else:
        models_to_check = MODEL_DATABASE

    # Check models against usable RAM (not current available)
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
        from tinybench.backends.ollama import OllamaBackend
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

