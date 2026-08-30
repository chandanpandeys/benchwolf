"""llama-cpp-python backend for BenchWolf."""

from __future__ import annotations

import time
from pathlib import Path

from benchwolf.backends.base import Backend, GenerationResult


class LlamaCppBackend(Backend):
    def __init__(self, n_ctx: int = 2048, n_gpu_layers: int = 0):
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._model = None
        self._model_path: str | None = None

    @property
    def name(self) -> str:
        return "llamacpp"

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def load_model(self, model_name: str) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install: pip install benchwolf[llamacpp]"
            ) from exc

        path = Path(model_name)
        if not path.exists():
            raise RuntimeError(f"Model file not found: {model_name}")
        if path.suffix.lower() != ".gguf":
            raise RuntimeError(f"Expected a .gguf model file, got: {path.suffix}")

        try:
            self._model = Llama(
                model_path=str(path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc
        self._model_path = str(path)

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        wall_start = time.perf_counter()
        first_token_time: float | None = None
        generated: list[str] = []
        completion_tokens = 0

        for chunk in self._model(prompt, max_tokens=max_tokens, stream=True, echo=False):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated.append(chunk["choices"][0].get("text", ""))
            completion_tokens += 1

        wall_end = time.perf_counter()
        total_s = wall_end - wall_start
        ttft_s = (first_token_time - wall_start) if first_token_time else total_s
        generation_s = max(total_s - ttft_s, 0.001)
        prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))

        return GenerationResult(
            text="".join(generated),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_eval_duration_s=ttft_s,
            generation_duration_s=generation_s,
            ttft_s=ttft_s,
            total_duration_s=total_s,
        )

    def get_model_info(self) -> dict:
        info: dict = {"backend": "llamacpp", "model_path": self._model_path or "none"}
        if self._model is None:
            return info

        info["context_size"] = self._n_ctx
        info["gpu_layers"] = self._n_gpu_layers
        metadata = getattr(self._model, "metadata", {}) or {}
        if self._model_path:
            upper_name = Path(self._model_path).stem.upper()
            quants = (
                "Q2_K",
                "Q3_K_S",
                "Q3_K_M",
                "Q3_K_L",
                "Q4_0",
                "Q4_K_S",
                "Q4_K_M",
                "Q5_0",
                "Q5_K_S",
                "Q5_K_M",
                "Q6_K",
                "Q8_0",
                "F16",
                "F32",
            )
            info["quantization"] = next((q for q in quants if q in upper_name), "unknown")
        if "general.parameter_count" in metadata:
            params = int(metadata["general.parameter_count"])
            info["parameter_size"] = (
                f"{params / 1_000_000_000:.1f}B"
                if params >= 1_000_000_000
                else f"{params / 1_000_000:.0f}M"
            )
        return info

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = None
