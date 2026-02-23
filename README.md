# 🔬 TinyBench

**Edge AI Benchmark Tool — Measure LLM performance on any hardware in one command.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Quick Start

```bash
pip install -e .
tinybench preflight          # Check what your hardware can run (no downloads needed)
tinybench run --model qwen2.5:3b   # Benchmark a model
```

TinyBench detects your hardware, runs speed/memory/power/quality benchmarks, and gives you a single **Edge Score** out of 100.

## What It Measures

| Category | Metrics |
|:---|:---|
| 🏎️ **Speed** | Tok/s (generation), tok/s (prompt eval), TTFT, sustained performance, thermal throttling |
| 💾 **Memory** | Peak RAM, model footprint, RAM utilization % |
| ⚡ **Power** | Watts consumed, tok/s per watt, energy per token (mJ) |
| 🎯 **Quality** | mini-MMLU (100 questions), mini-HumanEval (20 coding problems) |

## Commands

```bash
# Check if your hardware can run LLMs (no Ollama needed!)
tinybench preflight

# Check a specific model
tinybench preflight --model mistral:7b

# Full benchmark
tinybench run --model qwen2.5-coder:3b

# Speed-only (fast, ~2 min)
tinybench run --model qwen2.5:3b --only speed

# Compare models side-by-side
tinybench compare --models "qwen2.5:3b,phi3:3.8b"

# View your benchmark history
tinybench leaderboard

# Show hardware info + backend status
tinybench info

# Export results
tinybench run --model qwen2.5:3b --export json -o results.json
tinybench run --model qwen2.5:3b --export markdown -o results.md
```

## Preflight Check

Don't know which model to run? `tinybench preflight` checks your hardware and tells you — **without downloading anything**:

```
📊 Model Compatibility:

 Model               RAM Needed   Status
─────────────────────────────────────────
 qwen2.5:0.5b (Q4)    0.7 GB     ✅ Easy — will run smoothly
 phi3:mini (Q4)        2.4 GB     ✅ Easy — will run smoothly
 llama3.1:8b (Q4)      4.5 GB     ✅ Good — enough headroom
 deepseek-r1:14b       7.5 GB     ❌ Won't fit — not enough RAM

💡 Recommended: Start with llama3.1:8b
```

## Requirements

- **Python 3.10+**
- **Ollama** running locally ([install](https://ollama.ai)) — *for benchmarks only*
- A model pulled: `ollama pull qwen2.5:3b`

> **Note:** `tinybench preflight` and `tinybench info` work without Ollama.

## Installation

```bash
# From source
git clone https://github.com/AgenticBitBox/tinybench.git
cd tinybench
pip install -e .

# With llama.cpp support
pip install -e ".[llamacpp]"
```

## Sample Output

```
╭──────────────────────────────────────────────╮
│           🔬 TinyBench v0.1.0                │
│          Edge AI Benchmark Tool              │
╰──────────────────────────────────────────────╯

📦 Model: qwen2.5:3b (Q4_K_M)
💻 Hardware: AMD Ryzen 7 | x86_64 | 16.0GB RAM
🔌 Backend: ollama

┌──────────────────┬────────────┬─────────┐
│ Metric           │ Value      │ Rating  │
├──────────────────┼────────────┼─────────┤
│ Tok/s (gen)      │ 24.3       │ ⭐⭐⭐⭐⭐ │
│ TTFT             │ 0.42s      │ ⭐⭐⭐⭐⭐ │
│ Peak RAM         │ 2.8 GB     │ ⭐⭐⭐⭐  │
│ Power            │ 12.1 W     │ ⭐⭐⭐   │
│ Tok/s/W          │ 2.01       │ ⭐⭐⭐⭐  │
│ mini-MMLU (100q) │ 62.0%      │ ⭐⭐⭐⭐  │
│ mini-HumanEval   │ 40.0%      │ ⭐⭐⭐⭐  │
└──────────────────┴────────────┴─────────┘

🏆 Edge Score: 74/100
```

## Leaderboard

Every benchmark run is automatically saved to your local leaderboard:

```bash
tinybench leaderboard          # View all results
tinybench leaderboard --clear  # Clear history
```

Results are stored in `~/.tinybench/results/` as JSON files.

## Power Measurement

TinyBench uses the best available method:

| Tier | Method | Platform | Accuracy |
|:---|:---|:---|:---|
| 1 | Intel RAPL | Intel/AMD x86 Linux | High |
| 2 | hwmon sysfs | ARM Linux | Medium |
| 3 | Battery drain | Any laptop | Medium |
| 4 | TDP estimation | Any | Low (estimated) |

## Cross-Platform

TinyBench is pure Python — works on **Windows**, **Linux**, and **macOS** out of the box. Power measurement is even more accurate on Linux (RAPL/hwmon access).

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

Made with 🔬 by [Agentic BitBox](https://github.com/AgenticBitBox)
