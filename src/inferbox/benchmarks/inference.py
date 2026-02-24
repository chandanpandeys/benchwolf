"""Inference speed and throughput benchmarks."""

from __future__ import annotations

import statistics
import time
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from inferbox.backends.base import Backend, GenerationResult
from inferbox.config import (
    BENCHMARK_RUNS,
    MAX_TOKENS_PER_RUN,
    SPEED_PROMPT,
    SUSTAINED_DURATION_SECONDS,
    SUSTAINED_PROMPT,
    THROTTLE_WINDOW_SECONDS,
    WARMUP_RUNS,
)
from inferbox.models import SpeedResult


def run_speed_benchmark(
    backend: Backend,
    runs: int = BENCHMARK_RUNS,
    max_tokens: int = MAX_TOKENS_PER_RUN,
    include_sustained: bool = True,
    progress: Optional[Progress] = None,
) -> SpeedResult:
    """Run inference speed benchmarks.

    Protocol:
      1. Warmup runs (discarded)
      2. Standard runs (averaged)
      3. Sustained run (throttle detection)
    """
    tok_s_values: list[float] = []
    tok_s_prompt_values: list[float] = []
    ttft_values: list[float] = []
    total_tokens = 0

    # --- Phase 1: Warmup ---
    if progress:
        task = progress.add_task("[yellow]Warmup...", total=WARMUP_RUNS)

    for i in range(WARMUP_RUNS):
        backend.generate(SPEED_PROMPT, max_tokens=max_tokens)
        if progress:
            progress.advance(task)

    # --- Phase 2: Standard Runs ---
    if progress:
        task = progress.add_task("[cyan]Speed benchmark...", total=runs)

    for i in range(runs):
        result = backend.generate(SPEED_PROMPT, max_tokens=max_tokens)

        tok_s_values.append(result.tok_s_generation)
        if result.tok_s_prompt > 0:
            tok_s_prompt_values.append(result.tok_s_prompt)
        ttft_values.append(result.ttft_s)
        total_tokens += result.completion_tokens

        if progress:
            progress.advance(task)

    # --- Phase 3: Sustained Run (Throttle Detection) ---
    sustained_tok_s: Optional[float] = None
    throttle_pct: Optional[float] = None

    if include_sustained:
        if progress:
            task = progress.add_task("[magenta]Sustained test...", total=1)

        sustained_tok_s, throttle_pct = _run_sustained_benchmark(
            backend, max_tokens=max_tokens
        )

        if progress:
            progress.advance(task)

    return SpeedResult(
        tok_s_generation=statistics.mean(tok_s_values) if tok_s_values else 0.0,
        tok_s_prompt=statistics.mean(tok_s_prompt_values) if tok_s_prompt_values else 0.0,
        ttft_seconds=statistics.mean(ttft_values) if ttft_values else 0.0,
        total_tokens=total_tokens,
        runs=runs,
        sustained_tok_s=sustained_tok_s,
        throttle_percent=throttle_pct,
    )


def _run_sustained_benchmark(
    backend: Backend,
    max_tokens: int = 512,
) -> tuple[Optional[float], Optional[float]]:
    """Run sustained benchmark to detect thermal throttling.

    Strategy: run multiple short generations and track tok/s over time.
    Compare first window vs last window to detect degradation.
    """
    tok_s_over_time: list[tuple[float, float]] = []  # (elapsed_time, tok_s)
    start_time = time.time()
    deadline = start_time + SUSTAINED_DURATION_SECONDS

    iteration = 0
    while time.time() < deadline:
        result = backend.generate(SUSTAINED_PROMPT, max_tokens=max_tokens)
        elapsed = time.time() - start_time
        tok_s_over_time.append((elapsed, result.tok_s_generation))
        iteration += 1

        # At least 4 data points needed
        if iteration >= 8:
            break

    if len(tok_s_over_time) < 4:
        return None, None

    # Calculate average tok/s over the full sustained run
    all_tok_s = [t[1] for t in tok_s_over_time]
    sustained_avg = statistics.mean(all_tok_s)

    # Compare first half vs second half for throttle detection
    mid = len(all_tok_s) // 2
    first_half = statistics.mean(all_tok_s[:mid])
    second_half = statistics.mean(all_tok_s[mid:])

    if first_half > 0:
        throttle_pct = ((first_half - second_half) / first_half) * 100
    else:
        throttle_pct = 0.0

    return sustained_avg, round(throttle_pct, 1)
