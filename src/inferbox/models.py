"""Pydantic data models for InferBox results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

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
    gpu_name: Optional[str] = None
    npu_name: Optional[str] = None
    storage_type: Optional[str] = None
    memory_bandwidth_gb_s: Optional[float] = Field(None, description="Estimated memory bandwidth in GB/s")

    @property
    def fingerprint(self) -> str:
        """Generate a unique hardware fingerprint."""
        key = f"{self.cpu_name}|{self.cpu_arch}|{self.cpu_cores_physical}|{self.ram_total_gb:.1f}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def summary(self) -> str:
        """One-line hardware summary."""
        parts = [self.cpu_name, f"{self.cpu_arch}", f"{self.ram_total_gb:.1f}GB RAM"]
        if self.gpu_name:
            parts.append(self.gpu_name)
        if self.npu_name:
            parts.append(self.npu_name)
        return " | ".join(parts)


class SpeedResult(BaseModel):
    """Inference speed benchmark results."""

    tok_s_generation: float = Field(description="Tokens per second (generation)")
    tok_s_prompt: float = Field(0.0, description="Tokens per second (prompt eval)")
    ttft_seconds: float = Field(description="Time to first token in seconds")
    total_tokens: int = Field(description="Total tokens generated across all runs")
    runs: int = Field(description="Number of benchmark runs")
    sustained_tok_s: Optional[float] = Field(None, description="Sustained tok/s over long run")
    throttle_percent: Optional[float] = Field(None, description="Speed degradation percentage")
    thinking_tokens: Optional[int] = Field(None, description="Internal reasoning/thinking tokens generated")
    thinking_duration_s: Optional[float] = Field(None, description="Time spent emitting thinking tokens in seconds")


class MemoryResult(BaseModel):
    """Memory usage benchmark results."""

    model_config = ConfigDict(protected_namespaces=())

    peak_ram_mb: float = Field(description="Peak RAM usage during inference in MB")
    baseline_ram_mb: float = Field(description="RAM usage before model load in MB")
    model_ram_mb: float = Field(description="RAM used by model (peak - baseline)")
    ram_utilization_pct: float = Field(description="Percentage of total RAM used at peak")


class PowerResult(BaseModel):
    """Power consumption benchmark results."""

    method: str = Field(description="Measurement method (rapl/hwmon/battery/estimate)")
    avg_power_watts: float = Field(description="Average power during inference in watts")
    tok_s_per_watt: float = Field(description="Tokens per second per watt")
    energy_per_token_mj: float = Field(description="Energy per token in millijoules")
    measurement_duration_s: float = Field(description="Duration of power measurement")


class QualityResult(BaseModel):
    """Quality benchmark results."""

    mmlu_accuracy: Optional[float] = Field(None, description="Mini-MMLU accuracy (0-100)")
    mmlu_total: Optional[int] = Field(None, description="Total MMLU questions attempted")
    humaneval_pass_rate: Optional[float] = Field(None, description="Mini-HumanEval pass@1 (0-100)")
    humaneval_total: Optional[int] = Field(None, description="Total HumanEval problems attempted")


class BenchmarkResult(BaseModel):
    """Complete benchmark result."""

    model_config = ConfigDict(protected_namespaces=())

    inferbox_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str
    model_quantization: Optional[str] = None
    backend: str
    hardware: HardwareProfile
    speed: Optional[SpeedResult] = None
    memory: Optional[MemoryResult] = None
    power: Optional[PowerResult] = None
    quality: Optional[QualityResult] = None
    edge_score: Optional[int] = Field(None, description="Composite score 0-100")

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "BenchmarkResult":
        """Deserialize from JSON string."""
        return cls.model_validate_json(data)
