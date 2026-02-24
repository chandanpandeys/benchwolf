"""Tests for benchmark data and models."""

import json
from pathlib import Path

from inferbox.config import DATA_DIR
from inferbox.models import (
    BenchmarkResult,
    HardwareProfile,
    MemoryResult,
    PowerResult,
    QualityResult,
    SpeedResult,
)


def test_mmlu_data_loads():
    """mini-MMLU dataset should load and have 100 questions."""
    path = Path(DATA_DIR) / "mmlu_mini.json"
    assert path.exists(), f"Missing data file: {path}"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 100
    for item in data:
        assert "question" in item
        assert "choices" in item
        assert "answer" in item
        assert len(item["choices"]) == 4
        assert item["answer"] in ("A", "B", "C", "D")


def test_humaneval_data_loads():
    """mini-HumanEval dataset should load and have 20 problems."""
    path = Path(DATA_DIR) / "humaneval_mini.json"
    assert path.exists(), f"Missing data file: {path}"

    with open(path) as f:
        data = json.load(f)

    assert len(data) == 20
    for item in data:
        assert "task_id" in item
        assert "prompt" in item
        assert "tests" in item


def test_benchmark_result_json_roundtrip():
    """BenchmarkResult should serialize to JSON and back."""
    hw = HardwareProfile(
        cpu_name="Test CPU",
        cpu_arch="x86_64",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        cpu_freq_mhz=3000.0,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        os_name="Linux",
        os_version="6.1",
        python_version="3.10.0",
    )

    result = BenchmarkResult(
        inferbox_version="0.1.0",
        model_name="test-model:3b",
        model_quantization="Q4_K_M",
        backend="ollama",
        hardware=hw,
        speed=SpeedResult(
            tok_s_generation=15.0,
            tok_s_prompt=30.0,
            ttft_seconds=0.5,
            total_tokens=1000,
            runs=5,
            sustained_tok_s=14.0,
            throttle_percent=-3.2,
        ),
        memory=MemoryResult(
            peak_ram_mb=3000.0,
            baseline_ram_mb=500.0,
            model_ram_mb=2500.0,
            ram_utilization_pct=45.0,
        ),
        power=PowerResult(
            method="estimate",
            avg_power_watts=12.0,
            tok_s_per_watt=1.25,
            energy_per_token_mj=800.0,
            measurement_duration_s=30.0,
        ),
        quality=QualityResult(
            mmlu_accuracy=62.0,
            mmlu_total=100,
            humaneval_pass_rate=35.0,
            humaneval_total=20,
        ),
        edge_score=72,
    )

    json_str = result.to_json()
    restored = BenchmarkResult.from_json(json_str)

    assert restored.model_name == "test-model:3b"
    assert restored.speed.tok_s_generation == 15.0
    assert restored.memory.model_ram_mb == 2500.0
    assert restored.power.method == "estimate"
    assert restored.quality.mmlu_accuracy == 62.0
    assert restored.edge_score == 72
    assert restored.hardware.cpu_name == "Test CPU"
    assert restored.hardware.fingerprint == hw.fingerprint


def test_speed_result_fields():
    """SpeedResult should accept all expected fields."""
    s = SpeedResult(
        tok_s_generation=10.0,
        ttft_seconds=1.0,
        total_tokens=500,
        runs=3,
    )
    assert s.tok_s_prompt == 0.0  # default
    assert s.sustained_tok_s is None  # optional
    assert s.throttle_percent is None  # optional


def test_hardware_fingerprint_changes():
    """Different hardware configs should produce different fingerprints."""
    hw1 = HardwareProfile(cpu_name="CPU A", cpu_arch="x86_64", cpu_cores_physical=4, ram_total_gb=16.0)
    hw2 = HardwareProfile(cpu_name="CPU B", cpu_arch="x86_64", cpu_cores_physical=4, ram_total_gb=16.0)
    assert hw1.fingerprint != hw2.fingerprint
