"""Ollama HTTP backend for BenchWolf."""

from __future__ import annotations

import json
import re
import time

import requests

from benchwolf.backends.base import Backend, GenerationResult
from benchwolf.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT


class OllamaBackend(Backend):
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url.rstrip("/")
        self._model_name: str | None = None
        self._model_info: dict = {}

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self._base_url}/api/version", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def load_model(self, model_name: str) -> None:
        try:
            response = requests.post(
                f"{self._base_url}/api/show",
                json={"name": model_name},
                timeout=30,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError("Cannot connect to Ollama. Start it with: ollama serve") from exc

        if response.status_code != 200:
            raise RuntimeError(f"Model '{model_name}' not found. Run: ollama pull {model_name}")

        details = response.json().get("details", {})
        self._model_name = model_name
        self._model_info = {
            "name": model_name,
            "family": details.get("family", "unknown"),
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization": details.get("quantization_level", "unknown"),
            "format": details.get("format", "unknown"),
        }

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        if not self._model_name:
            raise RuntimeError("No model loaded. Call load_model() first.")

        wall_start = time.perf_counter()
        first_token_time: float | None = None
        generated: list[str] = []
        streamed_chunks = 0

        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
            },
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

        final_data: dict = {}
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done", False):
                final_data = chunk
                continue
            generated.append(chunk.get("response", ""))
            streamed_chunks += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()

        wall_end = time.perf_counter()
        full_text = "".join(generated)
        prompt_eval_s = final_data.get("prompt_eval_duration", 0) / 1e9
        eval_s = final_data.get("eval_duration", 0) / 1e9
        prompt_tokens = int(final_data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(final_data.get("eval_count", streamed_chunks) or streamed_chunks)
        total_s = wall_end - wall_start
        ttft_s = (first_token_time - wall_start) if first_token_time else total_s
        generation_s = eval_s or max(total_s - ttft_s, 0.001)

        thinking_tokens = None
        match = re.search(r"<think>(.*?)(?:</think>|$)", full_text, re.DOTALL)
        if match:
            thinking_tokens = max(len(match.group(1).split()), 1)

        return GenerationResult(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_eval_duration_s=prompt_eval_s,
            generation_duration_s=generation_s,
            ttft_s=ttft_s,
            total_duration_s=total_s,
            thinking_tokens=thinking_tokens,
        )

    def get_model_info(self) -> dict:
        return self._model_info

    def unload(self) -> None:
        if not self._model_name:
            return
        try:
            requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model_name, "keep_alive": 0},
                timeout=10,
            )
        except requests.RequestException:
            pass
