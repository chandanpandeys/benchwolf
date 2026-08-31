"""Backend interface shared by BenchWolf inference adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_eval_duration_s: float
    generation_duration_s: float
    ttft_s: float
    total_duration_s: float
    thinking_tokens: int | None = None
    thinking_duration_s: float | None = None

    @property
    def tok_s_generation(self) -> float:
        if self.generation_duration_s <= 0:
            return 0.0
        return self.completion_tokens / self.generation_duration_s

    @property
    def tok_s_prompt(self) -> float:
        if self.prompt_eval_duration_s <= 0:
            return 0.0
        return self.prompt_tokens / self.prompt_eval_duration_s


class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def load_model(self, model_name: str) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult: ...

    @abstractmethod
    def get_model_info(self) -> dict: ...

    def unload(self) -> None:
        pass
