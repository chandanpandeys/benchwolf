# Changelog

All notable changes to InferBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-30

### Added
- **Speed benchmarks** — Independent generation throughput (tok/s), prompt evaluation throughput (tok/s), TTFT (seconds), sustained performance over long runs, and thermal throttle detection.
- **Reasoning token telemetry** — Native extraction and duration tracking of `<think>...</think>` reasoning tokens (DeepSeek R1, QwQ).
- **Memory Bandwidth Ceiling Estimation** — Preflight calculates the theoretical peak speed of your hardware's memory bus (e.g. Apple Silicon Unified Memory, DDR5, LPDDR5X, Strix Halo 256-bit, Snapdragon X) vs. expected inference throughput.
- **MoE Model Architecture Support** — Accurately distinguishes between total weight footprint (for RAM allocation) and active parameter count (for memory-bandwidth bound generation speed).
- **Expanded 38+ Model Database** — Pre-computed architectural sizing for Qwen 2.5 & 3, Llama 3.x & 4 (Scout), DeepSeek R1 & V3 (671B MoE), Phi-3.5 & 4, Gemma 2 & 3, SmolLM2, and Mistral NeMo.
- **Modern Hardware Profiling** — Cross-platform detection for Apple Silicon M1-M4 (Pro/Max/Ultra), Qualcomm Snapdragon X Elite/Plus, AMD Ryzen AI 300 / Max 300 (XDNA 2 NPUs), and Intel Lunar Lake / Arrow Lake (Core Ultra).
- **Multi-tier Power Measurement** — 4-tier system: Intel RAPL (Linux), hwmon sysfs (ARM), smart battery sensor telemetry, and dynamic CPU-TDP utilization scaling.
- **Quality Benchmarks** — mini-MMLU (100 multi-subject questions) and mini-HumanEval (20 sandboxed coding problems).
- **Composite Edge Score** — Single 0–100 score synthesizing speed, memory efficiency, power conservation, and model intelligence.
- **Preflight Check** — Immediate hardware compatibility and throughput estimation without downloading any model files.
- **Model Comparison & Leaderboard** — Side-by-side benchmarking and local persistent history in `~/.inferbench/results/`.
- **Automated CI/CD** — GitHub Actions test matrix across Windows, Linux, and macOS on Python 3.10, 3.11, and 3.12, with automated PyPI publishing.

### Fixed
- **MMLU Option Parser Bug** — Fixed regex matching error that incorrectly extracted leading letters from conversational preambles. Replaced with robust multi-strategy parsing.
- **Battery Sensor Precision** — Upgraded battery telemetry to read high-precision Linux `/sys/class/power_supply` micro-watt sensors and calculate discharge rates on Windows/macOS.
- **Windows CPU Detection** — Added modern PowerShell `Get-CimInstance` support to eliminate dependency on deprecated `wmic`.
