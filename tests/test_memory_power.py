from benchwolf.backends.base import Backend, GenerationResult
from benchwolf.benchmarks.memory import run_memory_benchmark
from benchwolf.models import PowerResult


class FakeBackend(Backend):
    @property
    def name(self):
        return "fake"

    def is_available(self):
        return True

    def load_model(self, model_name):
        pass

    def generate(self, prompt, max_tokens=256):
        return GenerationResult("ok", 10, 10, 0.1, 1.0, 0.1, 1.1)

    def get_model_info(self):
        return {}


def test_memory_result_uses_system_pressure_terms():
    result = run_memory_benchmark(FakeBackend(), max_tokens=1, sample_interval_s=0.01)
    assert result.peak_system_ram_mb >= result.baseline_system_ram_mb
    assert result.inference_delta_mb >= 0
    assert result.sample_count >= 2


def test_power_result_allows_unavailable_values():
    result = PowerResult(
        method="rapl",
        source="unavailable",
        measurement_duration_s=1.0,
        sample_count=0,
    )
    assert result.avg_power_watts is None
    assert result.tok_s_per_watt is None
