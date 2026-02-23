"""Tests for backend interface and availability checks."""

from tinybench.backends.base import Backend, GenerationResult
from tinybench.backends.ollama import OllamaBackend


def test_generation_result_tok_s():
    """GenerationResult should correctly compute tok/s."""
    result = GenerationResult(
        text="hello world",
        prompt_tokens=10,
        completion_tokens=20,
        prompt_eval_duration_s=0.5,
        generation_duration_s=2.0,
        ttft_s=0.1,
        total_duration_s=2.5,
    )
    assert result.tok_s_generation == 10.0  # 20 / 2.0
    assert result.tok_s_prompt == 20.0  # 10 / 0.5


def test_generation_result_zero_duration():
    """Zero duration should return 0 tok/s, not crash."""
    result = GenerationResult(
        text="",
        prompt_tokens=0,
        completion_tokens=0,
        prompt_eval_duration_s=0.0,
        generation_duration_s=0.0,
        ttft_s=0.0,
        total_duration_s=0.0,
    )
    assert result.tok_s_generation == 0.0
    assert result.tok_s_prompt == 0.0


def test_ollama_backend_interface():
    """OllamaBackend should implement all Backend methods."""
    backend = OllamaBackend()
    assert isinstance(backend, Backend)
    assert backend.name == "ollama"
    # is_available may return True or False depending on system
    assert isinstance(backend.is_available(), bool)


def test_llamacpp_backend_interface():
    """LlamaCppBackend should implement Backend interface."""
    try:
        from tinybench.backends.llamacpp import LlamaCppBackend
        backend = LlamaCppBackend()
        assert isinstance(backend, Backend)
        assert backend.name == "llamacpp"
        assert isinstance(backend.is_available(), bool)
    except ImportError:
        # llama-cpp-python not installed, skip
        pass
