"""BenchWolf Edge Score v1 methodology."""

from __future__ import annotations

from dataclasses import dataclass

from benchwolf.models import BenchmarkResult


@dataclass(frozen=True)
class ScoreSummary:
    score: int | None
    partial: bool
    available_weight: int
    full_weight: int = 100


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def calculate_edge_score(result: BenchmarkResult) -> ScoreSummary:
    """Calculate Edge Score v1 and report whether it is partial.

    v1 weights:
      generation throughput 35
      TTFT 10
      sustained/throttle stability 5
      system RAM headroom 20
      measured power efficiency 15
      mini-MMLU 15

    Optional HumanEval never changes the score because it requires executing
    model-generated code and is intentionally opt-in.
    """
    components: list[tuple[float, int]] = []

    if result.speed:
        components.append((_clamp(result.speed.tok_s_generation / 20.0 * 100.0), 35))
        ttft_score = 100.0 - ((result.speed.ttft_seconds - 0.25) / 3.75 * 100.0)
        components.append((_clamp(ttft_score), 10))
        if result.speed.throttle_percent is not None:
            stability = 100.0 - abs(result.speed.throttle_percent) * 5.0
            components.append((_clamp(stability), 5))

    if result.memory:
        components.append((_clamp(100.0 - result.memory.ram_utilization_pct), 20))

    if (
        result.power
        and result.power.source == "measured"
        and result.power.tok_s_per_watt is not None
    ):
        components.append((_clamp(result.power.tok_s_per_watt / 3.0 * 100.0), 15))

    if result.quality and result.quality.mmlu_accuracy is not None:
        components.append((_clamp(result.quality.mmlu_accuracy), 15))

    if not components:
        return ScoreSummary(score=None, partial=True, available_weight=0)

    available_weight = sum(weight for _, weight in components)
    weighted = sum(score * weight for score, weight in components)
    score = round(weighted / available_weight)
    return ScoreSummary(score=score, partial=available_weight < 100, available_weight=available_weight)
