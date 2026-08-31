from benchwolf.models import (
    BenchmarkResult,
    HardwareProfile,
    MemoryResult,
    PowerResult,
    QualityResult,
    SpeedResult,
)
from benchwolf.scoring import calculate_edge_score


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        model_name="demo",
        backend="test",
        hardware=HardwareProfile(ram_total_gb=16, ram_available_gb=8),
    )


def test_empty_score_is_partial_and_none():
    summary = calculate_edge_score(_result())
    assert summary.score is None
    assert summary.partial is True
    assert summary.available_weight == 0


def _populated_result(power_source: str) -> BenchmarkResult:
    result = _result()
    result.speed = SpeedResult(
        tok_s_generation=20,
        ttft_seconds=0.25,
        total_tokens=100,
        runs=2,
        sustained_tok_s=20,
        throttle_percent=0,
    )
    result.memory = MemoryResult(
        baseline_system_ram_mb=4000,
        peak_system_ram_mb=5000,
        inference_delta_mb=1000,
        ram_utilization_pct=40,
        sample_count=10,
    )
    result.power = PowerResult(
        method="rapl" if power_source == "measured" else "estimate",
        source=power_source,
        avg_power_watts=10,
        tok_s_per_watt=2,
        energy_per_token_mj=500,
        measurement_duration_s=1,
        sample_count=5,
    )
    result.quality = QualityResult(mmlu_accuracy=70, mmlu_total=100)
    return result


def test_estimated_power_does_not_count_toward_full_score():
    summary = calculate_edge_score(_populated_result("estimated"))
    assert summary.partial is True
    assert summary.available_weight == 85


def test_measured_power_can_complete_full_score():
    summary = calculate_edge_score(_populated_result("measured"))
    assert summary.partial is False
    assert summary.available_weight == 100
