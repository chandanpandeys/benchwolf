"""Rich console output for InferBox results."""

from __future__ import annotations

import io
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inferbox import __version__
from inferbox.config import RATING_THRESHOLDS
from inferbox.models import BenchmarkResult, HardwareProfile

# Force UTF-8 output on Windows to support emoji
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


console = Console(force_terminal=True)


def print_header() -> None:
    """Print InferBox header banner."""
    header = Text()
    header.append("🔬 InferBox", style="bold bright_cyan")
    header.append(f" v{__version__}", style="dim")
    header.append("\n")
    header.append("Edge AI Benchmark Tool", style="italic")

    console.print(Panel(header, border_style="bright_cyan", expand=False))
    console.print()


def print_hardware_info(hw: HardwareProfile) -> None:
    """Print hardware information table."""
    table = Table(title="💻 Hardware Profile", border_style="blue", show_lines=False)
    table.add_column("Property", style="bold")
    table.add_column("Value", style="bright_white")

    table.add_row("CPU", hw.cpu_name)
    table.add_row("Architecture", hw.cpu_arch)
    table.add_row("Cores", f"{hw.cpu_cores_physical}P / {hw.cpu_cores_logical}L")
    table.add_row("Frequency", f"{hw.cpu_freq_mhz:.0f} MHz")
    table.add_row("RAM Total", f"{hw.ram_total_gb:.1f} GB")
    table.add_row("RAM Available", f"{hw.ram_available_gb:.1f} GB")
    table.add_row("OS", f"{hw.os_name} {hw.os_version}")
    table.add_row("Python", hw.python_version)

    if hw.gpu_name:
        table.add_row("GPU", hw.gpu_name)
    if hw.npu_name:
        table.add_row("NPU", hw.npu_name)
    if hw.storage_type:
        table.add_row("Storage", hw.storage_type)

    table.add_row("Fingerprint", hw.fingerprint, style="dim")

    console.print(table)
    console.print()


def print_benchmark_results(result: BenchmarkResult) -> None:
    """Print complete benchmark results as a Rich table."""
    # Model info header
    model_text = Text()
    model_text.append(f"📦 Model: ", style="bold")
    model_text.append(result.model_name, style="bright_green")
    if result.model_quantization:
        model_text.append(f" ({result.model_quantization})", style="dim")
    model_text.append(f"\n💻 Hardware: ", style="bold")
    model_text.append(result.hardware.summary, style="bright_white")
    model_text.append(f"\n🔌 Backend: ", style="bold")
    model_text.append(result.backend, style="bright_white")
    console.print(Panel(model_text, border_style="green", expand=False))
    console.print()

    # Results table
    table = Table(
        title="📊 Benchmark Results",
        border_style="bright_cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=18)
    table.add_column("Value", justify="right", min_width=12)
    table.add_column("Rating", justify="center", min_width=8)

    # Speed results
    if result.speed:
        s = result.speed
        table.add_row(
            "Tok/s (gen)",
            f"{s.tok_s_generation:.1f}",
            _rating_stars("tok_s", s.tok_s_generation),
        )
        if s.tok_s_prompt > 0:
            table.add_row(
                "Tok/s (prompt)",
                f"{s.tok_s_prompt:.1f}",
                _rating_stars("tok_s", s.tok_s_prompt / 2.0),
            )
        table.add_row(
            "TTFT",
            f"{s.ttft_seconds:.2f}s",
            _rating_stars("ttft", s.ttft_seconds),
        )
        if s.thinking_tokens is not None and s.thinking_tokens > 0:
            table.add_row(
                "Reasoning Tokens",
                f"{s.thinking_tokens:,} tokens",
                "🧠",
            )
        if s.sustained_tok_s is not None:
            table.add_row(
                "Sustained Tok/s",
                f"{s.sustained_tok_s:.1f}",
                "",
            )
        if s.throttle_percent is not None:
            throttle_str = f"{s.throttle_percent:+.1f}%"
            table.add_row(
                "Throttle",
                throttle_str,
                _rating_stars("throttle_pct", abs(s.throttle_percent)),
            )

    # Memory results
    if result.memory:
        m = result.memory
        table.add_row(
            "Peak RAM",
            f"{m.peak_ram_mb / 1024:.1f} GB",
            _rating_stars("ram_gb", m.peak_ram_mb / 1024),
        )
        table.add_row(
            "Model RAM",
            f"{m.model_ram_mb / 1024:.1f} GB",
            "",
        )
        table.add_row(
            "RAM Usage",
            f"{m.ram_utilization_pct:.0f}%",
            "",
        )

    # Power results
    if result.power:
        p = result.power
        table.add_row(
            "Power",
            f"{p.avg_power_watts:.1f} W",
            _rating_stars("power_w", p.avg_power_watts),
        )
        table.add_row(
            "Tok/s/W",
            f"{p.tok_s_per_watt:.2f}",
            _rating_stars("tok_s_per_w", p.tok_s_per_watt),
        )
        table.add_row(
            "Energy/token",
            f"{p.energy_per_token_mj:.1f} mJ",
            "",
        )
        table.add_row(
            "Method",
            p.method,
            "",
            style="dim",
        )

    # Quality results
    if result.quality:
        q = result.quality
        if q.mmlu_accuracy is not None:
            table.add_row(
                f"mini-MMLU ({q.mmlu_total}q)",
                f"{q.mmlu_accuracy:.1f}%",
                _rating_stars("mmlu_pct", q.mmlu_accuracy),
            )
        if q.humaneval_pass_rate is not None:
            table.add_row(
                f"mini-HumanEval ({q.humaneval_total}p)",
                f"{q.humaneval_pass_rate:.1f}%",
                _rating_stars("humaneval_pct", q.humaneval_pass_rate),
            )

    console.print(table)
    console.print()

    # Edge Score
    if result.edge_score is not None:
        score = result.edge_score
        if score >= 80:
            style = "bold bright_green"
            emoji = "🏆"
        elif score >= 60:
            style = "bold bright_yellow"
            emoji = "⭐"
        elif score >= 40:
            style = "bold bright_red"
            emoji = "🔶"
        else:
            style = "bold red"
            emoji = "🔻"

        console.print(
            Panel(
                Text(f"{emoji} Edge Score: {score}/100", style=style, justify="center"),
                border_style="bright_cyan",
                expand=False,
            )
        )
        console.print()


def _rating_stars(metric: str, value: float) -> str:
    """Convert a metric value to star rating."""
    thresholds = RATING_THRESHOLDS.get(metric, [])
    if not thresholds:
        return ""

    # For "lower is better" metrics (ttft, ram, power, throttle)
    lower_is_better = metric in ("ttft", "ram_gb", "power_w", "throttle_pct")

    for threshold, stars in thresholds:
        if lower_is_better:
            if value <= threshold:
                return "⭐" * stars
        else:
            if value >= threshold:
                return "⭐" * stars

    return "⭐"


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print side-by-side comparison of multiple benchmark results."""
    table = Table(
        title="📊 Model Comparison",
        border_style="bright_magenta",
        show_lines=True,
    )

    table.add_column("Metric", style="bold", min_width=16)
    for r in results:
        table.add_column(r.model_name, justify="right", min_width=14)

    # Speed row
    tok_s_values = []
    for r in results:
        val = r.speed.tok_s_generation if r.speed else 0
        tok_s_values.append(val)

    best_tok_s = max(tok_s_values) if tok_s_values else 0
    tok_s_cells = []
    for v in tok_s_values:
        style = "bold bright_green" if v == best_tok_s else ""
        tok_s_cells.append(f"[{style}]{v:.1f}[/]" if style else f"{v:.1f}")
    table.add_row("Tok/s (gen)", *tok_s_cells)

    # TTFT row
    ttft_values = []
    for r in results:
        val = r.speed.ttft_seconds if r.speed else 999
        ttft_values.append(val)

    best_ttft = min(ttft_values) if ttft_values else 0
    ttft_cells = []
    for v in ttft_values:
        style = "bold bright_green" if v == best_ttft else ""
        ttft_cells.append(f"[{style}]{v:.2f}s[/]" if style else f"{v:.2f}s")
    table.add_row("TTFT", *ttft_cells)

    # RAM row
    ram_values = []
    for r in results:
        val = r.memory.model_ram_mb / 1024 if r.memory else 0
        ram_values.append(val)

    best_ram = min(v for v in ram_values if v > 0) if any(v > 0 for v in ram_values) else 0
    ram_cells = []
    for v in ram_values:
        style = "bold bright_green" if v == best_ram and v > 0 else ""
        ram_cells.append(f"[{style}]{v:.1f} GB[/]" if style else f"{v:.1f} GB")
    table.add_row("Model RAM", *ram_cells)

    # Edge Score row
    score_values = []
    for r in results:
        val = r.edge_score or 0
        score_values.append(val)

    best_score = max(score_values) if score_values else 0
    score_cells = []
    for v in score_values:
        style = "bold bright_green" if v == best_score else ""
        score_cells.append(f"[{style}]{v}/100[/]" if style else f"{v}/100")
    table.add_row("Edge Score", *score_cells)

    console.print(table)
    console.print()
