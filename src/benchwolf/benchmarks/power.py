"""Power-efficiency benchmark orchestration."""

from __future__ import annotations

import statistics

from benchwolf.backends.base import Backend
from benchwolf.config import MAX_TOKENS_PER_RUN, POWER_PROMPT
from benchwolf.hardware.power import PowerMeter
from benchwolf.models import PowerResult


def run_power_benchmark(
    backend: Backend,
    max_tokens: int = MAX_TOKENS_PER_RUN,
    runs: int = 3,
) -> PowerResult:
    """Measure or estimate power concurrently with real generation work."""
    meter = PowerMeter()
    rates: list[float] = []
    meter.start()
    try:
        for _ in range(max(1, runs)):
            generation = backend.generate(POWER_PROMPT, max_tokens=max_tokens)
            if generation.tok_s_generation > 0:
                rates.append(generation.tok_s_generation)
    finally:
        measurement = meter.stop()

    tok_s = statistics.mean(rates) if rates else None
    watts = measurement.avg_watts
    tok_s_per_watt = None
    energy_per_token_mj = None
    if tok_s and watts and watts > 0:
        tok_s_per_watt = tok_s / watts
        energy_per_token_mj = watts / tok_s * 1000.0

    return PowerResult(
        method=measurement.method,
        source=measurement.source,
        avg_power_watts=round(watts, 2) if watts is not None else None,
        tok_s_per_watt=round(tok_s_per_watt, 3) if tok_s_per_watt is not None else None,
        energy_per_token_mj=(
            round(energy_per_token_mj, 2) if energy_per_token_mj is not None else None
        ),
        measurement_duration_s=round(measurement.duration_s, 2),
        sample_count=len(measurement.samples),
    )
