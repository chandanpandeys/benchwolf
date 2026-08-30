"""Tests for benchmark data, models, and quality evaluation."""

import json
from pathlib import Path

from inferbench.benchmarks.quality import _extract_answer
from inferbench.config import DATA_DIR
from inferbench.data.model_db import (
    MODEL_DATABASE,
    get_model_spec,
    get_quant_ram_table,
)
from inferbench.models import (
    BenchmarkResult,
    HardwareProfile,
    MemoryResult,
    PowerResult,
    QualityResult,
    SpeedResult,
)
from inferbench.preflight import _estimate_tok_s, run_preflight


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

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 20
    for item in data:
        assert "task_id" in item
        assert "prompt" in item
        assert "tests" in item


def test_mmlu_extract_answer_various_formats():
    """_extract_answer should accurately parse diverse LLM response styles."""
    # Direct letter
    assert _extract_answer("A") == "A"
    assert _extract_answer("  b  ") == "B"
    assert _extract_answer("C.") == "C"
    assert _extract_answer("D)") == "D"

    # Preambles (Bug fix verification: must not match 'a' in 'answer' or 'b' in 'based')
    assert _extract_answer("The answer is D") == "D"
    assert _extract_answer("The correct answer is (B)") == "B"
    assert _extract_answer("Based on the choices, D is correct.") == "D"
    assert _extract_answer("Answer: C") == "C"
    assert _extract_answer("Option A is the right choice") == "A"
    assert _extract_answer("Choice: B") == "B"

    # Markdown / formatting
    assert _extract_answer("**A**") == "A"
    assert _extract_answer("(C)") == "C"
    assert _extract_answer("[D]") == "D"

    # Thinking models with <think> tags (DeepSeek R1 / QwQ)
    thinking_output = (
        "<think>\n"
        "Let's analyze choice A, B, C, and D.\n"
        "Option A is tempting, but option C has the correct formula.\n"
        "</think>\n"
        "The correct answer is C."
    )
    assert _extract_answer(thinking_output) == "C"

    # None cases
    assert _extract_answer("") is None
    assert _extract_answer("No valid option found") is None


def test_model_database_expanded_2026():
    """Model database should include 35+ models with MoE specifications."""
    assert len(MODEL_DATABASE) >= 35

    # Check key models
    qwen = get_model_spec("qwen2.5:3b")
    assert qwen is not None
    assert qwen.params_b == 3.0

    phi4 = get_model_spec("phi4:14b")
    assert phi4 is not None
    assert phi4.family == "Phi-4"

    # Check MoE models
    llama4 = get_model_spec("llama4:scout")
    assert llama4 is not None
    assert llama4.is_moe is True
    assert llama4.active_params_b == 12.0
    assert llama4.effective_inference_params_b == 12.0

    deepseek_v3 = get_model_spec("deepseek-v3:671b")
    assert deepseek_v3 is not None
    assert deepseek_v3.is_moe is True
    assert deepseek_v3.active_params_b == 37.0
    assert deepseek_v3.effective_inference_params_b == 37.0


def test_quant_ram_table():
    """Quantization RAM table should calculate valid estimates for various bit depths."""
    table = get_quant_ram_table(8.0)
    assert "Q4_K_M (4-bit default)" in table
    assert "Q8_0 / FP8 (8-bit)" in table
    assert "FP16 (16-bit unquantized)" in table
    # Higher bit depths should take strictly more RAM
    assert table["FP16 (16-bit unquantized)"] > table["Q8_0 / FP8 (8-bit)"] > table["Q4_K_M (4-bit default)"]


def test_preflight_bandwidth_ceiling_calculation():
    """Preflight check should compute memory bandwidth ceiling."""
    spec = get_model_spec("qwen2.5:3b")
    assert spec is not None

    hw = HardwareProfile(
        cpu_name="Apple M4 Pro",
        cpu_arch="arm64",
        cpu_cores_physical=12,
        ram_total_gb=36.0,
        memory_bandwidth_gb_s=273.0,
    )

    tok_s, ceiling = _estimate_tok_s(spec, hw)
    assert ceiling is not None
    assert ceiling > 100.0  # M4 Pro has 273 GB/s bandwidth -> very high peak ceiling
    assert tok_s > 20.0


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
        memory_bandwidth_gb_s=64.0,
        npu_name="AMD XDNA 2 NPU (50+ TOPS)",
    )

    result = BenchmarkResult(
        inferbench_version="0.1.0",
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
            thinking_tokens=120,
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
    assert restored.speed.thinking_tokens == 120
    assert restored.memory.model_ram_mb == 2500.0
    assert restored.power.method == "estimate"
    assert restored.quality.mmlu_accuracy == 62.0
    assert restored.edge_score == 72
    assert restored.hardware.cpu_name == "Test CPU"
    assert restored.hardware.npu_name == "AMD XDNA 2 NPU (50+ TOPS)"
    assert restored.hardware.memory_bandwidth_gb_s == 64.0
    assert restored.hardware.fingerprint == hw.fingerprint


def test_speed_result_fields():
    """SpeedResult should accept all expected fields."""
    s = SpeedResult(
        tok_s_generation=10.0,
        ttft_seconds=1.0,
        total_tokens=500,
        runs=3,
        thinking_tokens=50,
    )
    assert s.tok_s_prompt == 0.0  # default
    assert s.sustained_tok_s is None  # optional
    assert s.throttle_percent is None  # optional
    assert s.thinking_tokens == 50


def test_hardware_fingerprint_changes():
    """Different hardware configs should produce different fingerprints."""
    hw1 = HardwareProfile(cpu_name="CPU A", cpu_arch="x86_64", cpu_cores_physical=4, ram_total_gb=16.0)
    hw2 = HardwareProfile(cpu_name="CPU B", cpu_arch="x86_64", cpu_cores_physical=4, ram_total_gb=16.0)
    assert hw1.fingerprint != hw2.fingerprint
