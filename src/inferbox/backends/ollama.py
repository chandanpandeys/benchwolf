"""Ollama backend for InferBox.

Uses the Ollama HTTP API for inference and timing measurement.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests

from inferbox.backends.base import Backend, GenerationResult
from inferbox.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT


class OllamaBackend(Backend):
    """Ollama HTTP API backend."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url.rstrip("/")
        self._model_name: Optional[str] = None
        self._model_info: dict = {}

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = requests.get(f"{self._base_url}/api/version", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def load_model(self, model_name: str) -> None:
        """Verify model is available in Ollama."""
        self._model_name = model_name

        # Check if model exists locally
        try:
            resp = requests.post(
                f"{self._base_url}/api/show",
                json={"name": model_name},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._model_info = {
                    "name": model_name,
                    "family": data.get("details", {}).get("family", "unknown"),
                    "parameter_size": data.get("details", {}).get("parameter_size", "unknown"),
                    "quantization": data.get("details", {}).get("quantization_level", "unknown"),
                    "format": data.get("details", {}).get("format", "unknown"),
                }
                return
            else:
                raise RuntimeError(
                    f"Model '{model_name}' not found in Ollama. "
                    f"Run: ollama pull {model_name}"
                )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running? Start with: ollama serve"
            )

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text using Ollama streaming API and measure timing."""
        if not self._model_name:
            raise RuntimeError("No model loaded. Call load_model() first.")

        wall_start = time.perf_counter()
        first_token_time: Optional[float] = None
        generated_text = []
        completion_tokens = 0

        # Ollama streaming response
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.0,  # Deterministic for benchmarking
                },
            },
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()

        # Parse streaming NDJSON response
        final_data = {}
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)

            if not chunk.get("done", False):
                # Token received
                token = chunk.get("response", "")
                generated_text.append(token)
                completion_tokens += 1

                if first_token_time is None:
                    first_token_time = time.perf_counter()
            else:
                # Final message with stats
                final_data = chunk

        wall_end = time.perf_counter()
        full_text = "".join(generated_text)

        # Extract reasoning/thinking metrics if model emits <think>...</think> tags
        thinking_tokens = None
        thinking_duration_s = None
        if "<think>" in full_text:
            import re
            think_match = re.search(r"<think>(.*?)(?:</think>|$)", full_text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                # Approximate token count for thinking block (~0.75 words per token)
                thinking_tokens = max(len(think_content.split()), 1)

        # Extract Ollama's own timing (in nanoseconds)
        prompt_eval_ns = final_data.get("prompt_eval_duration", 0)
        eval_ns = final_data.get("eval_duration", 0)
        prompt_tokens = final_data.get("prompt_eval_count", 0)
        eval_count = final_data.get("eval_count", completion_tokens)

        # Convert nanoseconds to seconds
        prompt_eval_s = prompt_eval_ns / 1e9 if prompt_eval_ns else 0.0
        generation_s = eval_ns / 1e9 if eval_ns else (wall_end - (first_token_time or wall_start))
        total_s = wall_end - wall_start
        ttft_s = (first_token_time - wall_start) if first_token_time else total_s

        return GenerationResult(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=eval_count or completion_tokens,
            prompt_eval_duration_s=prompt_eval_s,
            generation_duration_s=generation_s,
            ttft_s=ttft_s,
            total_duration_s=total_s,
            thinking_tokens=thinking_tokens,
            thinking_duration_s=thinking_duration_s,
        )

    def get_model_info(self) -> dict:
        """Get loaded model information."""
        return self._model_info

    def unload(self) -> None:
        """Unload model from Ollama memory."""
        if self._model_name:
            try:
                requests.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._model_name, "keep_alive": 0},
                    timeout=10,
                )
            except Exception:
                pass
