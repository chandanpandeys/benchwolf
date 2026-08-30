"""Inference throughput, latency, and sustained-behavior benchmarks."""

from __future__ import annotations

import statistics
import time

from rich.progress import Progress

from benchwolf.backends.base import Backend
from benchwolf.config import (
    BENCHMARK_RUNS,
    MAX_TOKENS_PER_RUN,
    SPEED_PROMPT,
    SUSTAINED_DURATION_SECONDS,
    SUSTAINED_PROMPT,
    WARMUP_RUNS,
)
from benchwolf.models import SpeedResult


def run_speed_benchmark(
    backend: Backend,
    runs: int = BENCHMARK_RUNS,
    max_tokens: int = MAX_TOKENS_PER_RUN,
    include_sustained: bool = True,
    progress: Progress | None = None,
) -> SpeedResult:
    runs = max(1, runs)
    for _ in range(WARMUP_RUNS):
        backend.generate(SPEED_PROMPT, max_tokens=max_tokens)

    task = progress.add_task("[cyan]Speed benchmark...", total=runs) if progress else None
    generation_rates: list[float] = []
    prompt_rates: list[float] = []
    ttfts: list[float] = []
    thinking: list[int] = []
    total_tokens = 0

    for _ in range(runs):
        result = backend.generate(SPEED_PROMPT, max_tokens=max_tokens)
        generation_rates.append(result.tok_s_generation)
        if result.tok_s_prompt > 0:
            prompt_rates.append(result.tok_s_prompt)
        ttfts.append(result.ttft_s)
        if result.thinking_tokens is not None:
            thinking.append(result.thinking_tokens)
        total_tokens += result.completion_tokens
        if progress and task is not None:
            progress.advance(task)

    sustained = throttle = None
    if include_sustained:
        sustained, throttle = _run_sustained_benchmark(backend, max_tokens=max_tokens)

    return SpeedResult(
        tok_s_generation=statistics.mean(generation_rates),
        tok_s_prompt=statistics.mean(prompt_rates) if prompt_rates else 0.0,
        ttft_seconds=statistics.mean(ttfts),
        total_tokens=total_tokens,
        runs=runs,
        sustained_tok_s=sustained,
        throttle_percent=throttle,
        thinking_tokens=sum(thinking) if thinking else None,
    )


def _run_sustained_benchmark(
    backend: Backend,
    max_tokens: int = 512,
) -> tuple[float | None, float | None]:
    samples: list[float] = []
    start = time.perf_counter()
    deadline = start + SUSTAINED_DURATION_SECONDS

    while time.perf_counter() < deadline and len(samples) < 8:
        result = backend.generate(SUSTAINED_PROMPT, max_tokens=max_tokens)
        samples.append(result.tok_s_generation)

    if len(samples) < 4:
        return None, None

    mid = len(samples) // 2
    first = statistics.mean(samples[:mid])
    second = statistics.mean(samples[mid:])
    throttle = ((first - second) / first * 100.0) if first > 0 else 0.0
    return round(statistics.mean(samples), 2), round(throttle, 1)
