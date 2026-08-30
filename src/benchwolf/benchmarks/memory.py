"""System-memory pressure sampling during inference."""

from __future__ import annotations

import threading

import psutil

from benchwolf.backends.base import Backend
from benchwolf.config import MAX_TOKENS_PER_RUN, MEMORY_SAMPLE_INTERVAL_SECONDS, SPEED_PROMPT
from benchwolf.models import MemoryResult


def _system_used_mb() -> float:
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / (1024**2)


def run_memory_benchmark(
    backend: Backend,
    max_tokens: int = MAX_TOKENS_PER_RUN,
    sample_interval_s: float = MEMORY_SAMPLE_INTERVAL_SECONDS,
) -> MemoryResult:
    """Sample system-wide RAM while inference is running.

    This intentionally does not call the delta "model RAM": Ollama can run in a
    separate process and OS caches/background applications affect system memory.
    """
    interval = max(0.01, sample_interval_s)
    vm = psutil.virtual_memory()
    baseline = _system_used_mb()
    samples = [baseline]
    stop_event = threading.Event()

    def sampler() -> None:
        while not stop_event.is_set():
            samples.append(_system_used_mb())
            stop_event.wait(interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        backend.generate(SPEED_PROMPT, max_tokens=max_tokens)
        backend.generate(SPEED_PROMPT, max_tokens=max_tokens)
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, interval * 4))
        samples.append(_system_used_mb())

    peak = max(samples)
    total_mb = vm.total / (1024**2)
    return MemoryResult(
        baseline_system_ram_mb=round(baseline, 1),
        peak_system_ram_mb=round(peak, 1),
        inference_delta_mb=round(max(0.0, peak - baseline), 1),
        ram_utilization_pct=round((peak / total_mb) * 100.0, 1),
        sample_count=len(samples),
        sample_interval_ms=round(interval * 1000),
    )
