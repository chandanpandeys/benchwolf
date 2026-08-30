from benchwolf.backends.base import Backend, GenerationResult
from benchwolf.benchmarks.quality import _extract_answer, _test_code, run_quality_benchmark


class FakeBackend(Backend):
    @property
    def name(self):
        return "fake"

    def is_available(self):
        return True

    def load_model(self, model_name):
        pass

    def generate(self, prompt, max_tokens=256):
        return GenerationResult("A", 1, 1, 0.1, 0.1, 0.1, 0.2)

    def get_model_info(self):
        return {}


def test_extract_answer_common_formats():
    assert _extract_answer("A") == "A"
    assert _extract_answer("Answer: c") == "C"
    assert _extract_answer("<think>x</think> The answer is D") == "D"


def test_humaneval_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr("benchwolf.benchmarks.quality._run_mini_mmlu", lambda *args: (50.0, 100))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HumanEval must not run without explicit opt-in")

    monkeypatch.setattr("benchwolf.benchmarks.quality._run_mini_humaneval", fail_if_called)
    result = run_quality_benchmark(FakeBackend())
    assert result.humaneval_enabled is False
    assert result.humaneval_pass_rate is None


def test_code_runner_handles_simple_candidate():
    assert _test_code("def add(a, b):\n    ", "return a + b\n", "assert add(2, 3) == 5")
