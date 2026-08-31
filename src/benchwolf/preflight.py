"""Estimate whether known local LLMs fit before downloading them."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum

from benchwolf.data.model_db import MODEL_DATABASE, ModelSpec, get_model_spec
from benchwolf.hardware.detect import detect_hardware
from benchwolf.models import HardwareProfile


class FitStatus(Enum):
    EASY = "easy"
    GOOD = "good"
    TIGHT = "tight"
    NO_FIT = "no_fit"
    NO_DISK = "no_disk"


@dataclass
class ModelFitResult:
    model: ModelSpec
    status: FitStatus
    available_ram_gb: float
    ram_headroom_gb: float
    disk_ok: bool
    estimated_tok_s: float | None = None
    bandwidth_ceiling_tok_s: float | None = None
    reason: str = ""

    @property
    def status_emoji(self) -> str:
        return {
            FitStatus.EASY: "✅",
            FitStatus.GOOD: "✅",
            FitStatus.TIGHT: "⚠️",
            FitStatus.NO_FIT: "❌",
            FitStatus.NO_DISK: "💾",
        }[self.status]

    @property
    def status_label(self) -> str:
        return {
            FitStatus.EASY: "Easy",
            FitStatus.GOOD: "Good",
            FitStatus.TIGHT: "Tight",
            FitStatus.NO_FIT: "Does not fit",
            FitStatus.NO_DISK: "Insufficient disk",
        }[self.status]


@dataclass
class PreflightReport:
    hardware: HardwareProfile
    available_ram_gb: float
    usable_ram_gb: float
    free_disk_gb: float
    model_results: list[ModelFitResult]
    recommended_model: str | None = None
    can_run_any: bool = False
    ollama_running: bool = False


def _estimate_tok_s(model: ModelSpec, hardware: HardwareProfile) -> tuple[float, float | None]:
    """Return a heuristic throughput estimate and theoretical bandwidth ceiling."""
    active_params = model.effective_inference_params_b
    q4_weight_gb = max(active_params * 0.55, 0.08)
    ceiling = None
    if hardware.memory_bandwidth_gb_s:
        ceiling = round(hardware.memory_bandwidth_gb_s / q4_weight_gb, 1)

    accelerated = bool(hardware.gpu_name) or "apple" in hardware.cpu_name.lower()
    if accelerated and ceiling:
        return round(max(1.0, ceiling * 0.58), 1), ceiling

    size_factor = 3.0 / max(active_params, 0.1)
    core_factor = min(max(hardware.cpu_cores_physical, 1) / 4.0, 2.5)
    freq_factor = min(hardware.cpu_freq_mhz / 2500.0, 1.6) if hardware.cpu_freq_mhz else 1.0
    estimate = 6.0 * size_factor * core_factor * freq_factor
    if ceiling and estimate > ceiling:
        estimate = ceiling * 0.5
    return round(max(estimate, 0.5), 1), ceiling


def _check_model_fit(
    model: ModelSpec,
    hardware: HardwareProfile,
    usable_ram_gb: float,
    free_disk_gb: float,
) -> ModelFitResult:
    if free_disk_gb < model.disk_size_gb + 1.0:
        return ModelFitResult(
            model=model,
            status=FitStatus.NO_DISK,
            available_ram_gb=usable_ram_gb,
            ram_headroom_gb=usable_ram_gb - model.min_ram_gb,
            disk_ok=False,
            reason=f"Needs {model.disk_size_gb:.1f} GB download plus 1 GB headroom.",
        )

    headroom = usable_ram_gb - model.min_ram_gb
    if headroom < 0:
        status = FitStatus.NO_FIT
    elif headroom < 1.0:
        status = FitStatus.TIGHT
    elif headroom < 2.5:
        status = FitStatus.GOOD
    else:
        status = FitStatus.EASY

    estimate = ceiling = None
    if status != FitStatus.NO_FIT:
        estimate, ceiling = _estimate_tok_s(model, hardware)
    return ModelFitResult(
        model=model,
        status=status,
        available_ram_gb=usable_ram_gb,
        ram_headroom_gb=round(headroom, 1),
        disk_ok=True,
        estimated_tok_s=estimate,
        bandwidth_ceiling_tok_s=ceiling,
        reason=f"{headroom:.1f} GB estimated RAM headroom.",
    )


def run_preflight(specific_model: str | None = None) -> PreflightReport:
    hardware = detect_hardware()
    usable_ram = max(hardware.ram_total_gb - 2.0, 0.5)
    try:
        disk = shutil.disk_usage(os.path.expanduser("~"))
        free_disk = disk.free / (1024**3)
    except OSError:
        free_disk = 0.0

    if specific_model:
        match = get_model_spec(specific_model)
        models = [match] if match else MODEL_DATABASE
    else:
        models = MODEL_DATABASE

    results = [_check_model_fit(m, hardware, usable_ram, free_disk) for m in models]
    runnable = [r for r in results if r.status in {FitStatus.EASY, FitStatus.GOOD}]
    recommended = max(runnable, key=lambda item: item.model.params_b).model.name if runnable else None

    try:
        from benchwolf.backends.ollama import OllamaBackend

        ollama_running = OllamaBackend().is_available()
    except Exception:
        ollama_running = False

    return PreflightReport(
        hardware=hardware,
        available_ram_gb=hardware.ram_available_gb,
        usable_ram_gb=round(usable_ram, 1),
        free_disk_gb=round(free_disk, 1),
        model_results=results,
        recommended_model=recommended,
        can_run_any=any(r.status not in {FitStatus.NO_FIT, FitStatus.NO_DISK} for r in results),
        ollama_running=ollama_running,
    )
