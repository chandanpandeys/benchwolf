"""Abstract backend interface for TinyBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationResult:
    """Result from a single text generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_eval_duration_s: float  # Time to process prompt
    generation_duration_s: float  # Time to generate tokens
    ttft_s: float  # Time to first token
    total_duration_s: float  # Total wall-clock time

    @property
    def tok_s_generation(self) -> float:
        """Tokens per second for generation."""
        if self.generation_duration_s <= 0:
            return 0.0
        return self.completion_tokens / self.generation_duration_s

    @property
    def tok_s_prompt(self) -> float:
        """Tokens per second for prompt processing."""
        if self.prompt_eval_duration_s <= 0:
            return 0.0
        return self.prompt_tokens / self.prompt_eval_duration_s


class Backend(ABC):
    """Abstract base class for inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'ollama', 'llamacpp')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and running."""
        ...

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load/prepare a model for inference."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text and return timing metrics."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get info about the loaded model (name, quantization, size)."""
        ...

    def unload(self) -> None:
        """Unload the model (optional cleanup)."""
        pass
