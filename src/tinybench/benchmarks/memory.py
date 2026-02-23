"""Memory usage benchmarks."""

from __future__ import annotations

import time
from typing import Optional

import psutil

from tinybench.backends.base import Backend
from tinybench.config import MAX_TOKENS_PER_RUN, SPEED_PROMPT
from tinybench.models import MemoryResult


def run_memory_benchmark(
    backend: Backend,
    max_tokens: int = MAX_TOKENS_PER_RUN,
) -> MemoryResult:
    """Measure memory usage during inference.

    Strategy:
      1. Measure baseline RAM (before model interaction)
      2. Run inference while polling RAM usage
      3. Report peak RAM and model footprint
    """
    process = psutil.Process()
    system_memory = psutil.virtual_memory()

    # Baseline: system memory usage before inference
    baseline_system_mb = (system_memory.total - system_memory.available) / (1024**2)

    # Track peak during inference
    peak_mb = baseline_system_mb
    samples: list[float] = []

    # Start inference in a simple polling loop
    # We run generate and sample memory before/after and during if possible
    pre_mem = psutil.virtual_memory()
    baseline_available_mb = pre_mem.available / (1024**2)

    # Sample before
    samples.append((pre_mem.total - pre_mem.available) / (1024**2))

    # Run inference
    backend.generate(SPEED_PROMPT, max_tokens=max_tokens)

    # Sample after
    post_mem = psutil.virtual_memory()
    post_used_mb = (post_mem.total - post_mem.available) / (1024**2)
    samples.append(post_used_mb)

    # Run a second generation to get a stable peak
    backend.generate(SPEED_PROMPT, max_tokens=max_tokens)
    peak_mem = psutil.virtual_memory()
    peak_used_mb = (peak_mem.total - peak_mem.available) / (1024**2)
    samples.append(peak_used_mb)

    peak_mb = max(samples)
    total_ram_mb = system_memory.total / (1024**2)

    return MemoryResult(
        peak_ram_mb=round(peak_mb, 1),
        baseline_ram_mb=round(samples[0], 1),
        model_ram_mb=round(peak_mb - samples[0], 1),
        ram_utilization_pct=round((peak_mb / total_ram_mb) * 100, 1),
    )
