from benchwolf.backends.base import GenerationResult
from benchwolf.backends.ollama import OllamaBackend


def test_generation_rate_math():
    result = GenerationResult(
        text="ok",
        prompt_tokens=10,
        completion_tokens=20,
        prompt_eval_duration_s=0.5,
        generation_duration_s=2.0,
        ttft_s=0.2,
        total_duration_s=2.5,
    )
    assert result.tok_s_generation == 10.0
    assert result.tok_s_prompt == 20.0


def test_zero_duration_is_zero_rate():
    result = GenerationResult("", 0, 0, 0.0, 0.0, 0.0, 0.0)
    assert result.tok_s_generation == 0.0
    assert result.tok_s_prompt == 0.0


def test_ollama_availability_returns_bool():
    assert isinstance(OllamaBackend(base_url="http://127.0.0.1:1").is_available(), bool)
