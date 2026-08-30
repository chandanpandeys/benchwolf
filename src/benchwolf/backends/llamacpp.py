"""llama-cpp-python backend for InferBench.

Provides direct access to llama.cpp via Python bindings for
more precise timing without HTTP overhead.

Install: pip install inferbench[llamacpp]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from inferbench.backends.base import Backend, GenerationResult


class LlamaCppBackend(Backend):
    """Backend using llama-cpp-python for direct llama.cpp access."""

    def __init__(self, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """Initialize llama-cpp-python backend.

        Args:
            n_ctx: Context window size (default: 2048).
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only).
        """
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._model = None
        self._model_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "llamacpp"

    def is_available(self) -> bool:
        """Check if llama-cpp-python is installed."""
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self, model_name: str) -> None:
        """Load a GGUF model file.

        Args:
            model_name: Path to a .gguf model file.

        Raises:
            RuntimeError: If the model file doesn't exist or can't be loaded.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install inferbench[llamacpp]"
            )

        model_path = Path(model_name)
        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_name}")

        if not model_path.suffix == ".gguf":
            raise RuntimeError(
                f"Expected .gguf file, got: {model_path.suffix}. "
                "Download GGUF models from https://huggingface.co"
            )

        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            self._model_path = str(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text using llama-cpp-python.

        Returns timing data extracted directly from llama.cpp internals.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        wall_start = time.perf_counter()
        ttft_recorded = False
        ttft = 0.0
        generated_text = ""
        completion_tokens = 0

        # Use streaming for TTFT measurement
        stream = self._model(
            prompt,
            max_tokens=max_tokens,
            stream=True,
            echo=False,
        )

        for chunk in stream:
            if not ttft_recorded:
                ttft = time.perf_counter() - wall_start
                ttft_recorded = True

            token_text = chunk["choices"][0].get("text", "")
            generated_text += token_text
            completion_tokens += 1

        wall_end = time.perf_counter()
        total_duration = wall_end - wall_start

        # Prompt eval is the time before first token
        prompt_eval_duration = ttft if ttft_recorded else 0.0

        # Generation duration is total minus prompt eval
        generation_duration = total_duration - prompt_eval_duration

        # Tokenize prompt to get token count
        prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))

        return GenerationResult(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_eval_duration_s=prompt_eval_duration,
            generation_duration_s=max(generation_duration, 0.001),
            ttft_s=ttft,
            total_duration_s=total_duration,
        )

    def get_model_info(self) -> dict:
        """Get model metadata from the GGUF file."""
        info: dict = {
            "backend": "llamacpp",
            "model_path": self._model_path or "none",
        }

        if self._model is not None:
            metadata = getattr(self._model, "metadata", {}) or {}

            # Extract quantization from filename or metadata
            if self._model_path:
                path = Path(self._model_path)
                name = path.stem
                # Common GGUF naming: model-Q4_K_M.gguf
                for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
                           "Q4_0", "Q4_K_S", "Q4_K_M",
                           "Q5_0", "Q5_K_S", "Q5_K_M",
                           "Q6_K", "Q8_0", "F16", "F32"]:
                    if q in name.upper():
                        info["quantization"] = q
                        break

            info["context_size"] = self._n_ctx
            info["gpu_layers"] = self._n_gpu_layers

            # Try to get parameter size from metadata
            if "general.parameter_count" in metadata:
                params = int(metadata["general.parameter_count"])
                if params >= 1_000_000_000:
                    info["parameter_size"] = f"{params / 1_000_000_000:.1f}B"
                else:
                    info["parameter_size"] = f"{params / 1_000_000:.0f}M"

        return info

    def unload(self) -> None:
        """Free model memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = None
