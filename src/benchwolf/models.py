"""Typed result models used by BenchWolf."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HardwareProfile(BaseModel):
    """Detected hardware information."""

    cpu_name: str = "Unknown"
    cpu_arch: str = "Unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    os_name: str = "Unknown"
    os_version: str = "Unknown"
    python_version: str = "Unknown"
    gpu_name: str | None = None
    npu_name: str | None = None
    storage_type: str | None = None
    memory_bandwidth_gb_s: float | None = Field(
        None,
        description="Estimated peak memory bandwidth in GB/s.",
    )

    @property
    def fingerprint(self) -> str:
        key = f"{self.cpu_name}|{self.cpu_arch}|{self.cpu_cores_physical}|{self.ram_total_gb:.1f}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def summary(self) -> str:
        parts = [self.cpu_name, self.cpu_arch, f"{self.ram_total_gb:.1f}GB RAM"]
        if self.gpu_name:
            parts.append(self.gpu_name)
        if self.npu_name:
            parts.append(self.npu_name)
        return " | ".join(parts)


class SpeedResult(BaseModel):
    """Inference speed benchmark results."""

    tok_s_generation: float = Field(description="Generation tokens per second.")
    tok_s_prompt: float = Field(0.0, description="Prompt-evaluation tokens per second.")
    ttft_seconds: float = Field(description="Time to first token in seconds.")
    total_tokens: int = Field(description="Total generated tokens across measured runs.")
    runs: int = Field(description="Number of measured benchmark runs.")
    sustained_tok_s: float | None = None
    throttle_percent: float | None = Field(
        None,
        description="Heuristic throughput change between the first and second sustained windows.",
    )
    thinking_tokens: int | None = Field(
        None,
        description="Approximate reasoning-token proxy derived from tagged reasoning text.",
    )
    thinking_duration_s: float | None = None


class MemoryResult(BaseModel):
    """System-wide memory pressure observed while the model is generating."""

    model_config = ConfigDict(protected_namespaces=())

    baseline_system_ram_mb: float
    peak_system_ram_mb: float
    inference_delta_mb: float = Field(description="Peak system-used RAM minus the pre-inference baseline.")
    ram_utilization_pct: float
    sample_count: int
    sample_interval_ms: int = 50


class PowerResult(BaseModel):
    """Power measurement or estimate collected while inference is running."""

    method: str
    source: Literal["measured", "estimated", "unavailable"]
    avg_power_watts: float | None = None
    tok_s_per_watt: float | None = None
    energy_per_token_mj: float | None = None
    measurement_duration_s: float
    sample_count: int = 0


class QualityResult(BaseModel):
    """Lightweight quality benchmark results."""

    mmlu_accuracy: float | None = None
    mmlu_total: int | None = None
    humaneval_pass_rate: float | None = None
    humaneval_total: int | None = None
    humaneval_enabled: bool = False


class BenchmarkResult(BaseModel):
    """Complete BenchWolf benchmark result."""

    model_config = ConfigDict(protected_namespaces=())

    benchwolf_version: str = "0.1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str
    model_quantization: str | None = None
    backend: str
    hardware: HardwareProfile
    speed: SpeedResult | None = None
    memory: MemoryResult | None = None
    power: PowerResult | None = None
    quality: QualityResult | None = None
    edge_score: int | None = Field(None, description="BenchWolf Edge Score v1 (0-100).")
    edge_score_version: str = "1"
    edge_score_is_partial: bool = True

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "BenchmarkResult":
        return cls.model_validate_json(data)
