"""Rich terminal rendering for BenchWolf."""

from __future__ import annotations

import io
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from benchwolf import __version__
from benchwolf.models import BenchmarkResult, HardwareProfile

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()


def print_header() -> None:
    text = Text()
    text.append("🐺 BenchWolf", style="bold cyan")
    text.append(f" v{__version__}", style="dim")
    text.append("\nLocal LLM benchmarks on your own hardware", style="italic")
    console.print(Panel(text, border_style="cyan", expand=False))
    console.print()


def print_hardware_info(hardware: HardwareProfile) -> None:
    table = Table(title="Hardware", show_lines=False)
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("CPU", hardware.cpu_name)
    table.add_row("Architecture", hardware.cpu_arch)
    table.add_row("Cores", f"{hardware.cpu_cores_physical}P / {hardware.cpu_cores_logical}L")
    table.add_row("RAM", f"{hardware.ram_total_gb:.1f} GB")
    table.add_row("OS", f"{hardware.os_name} {hardware.os_version}")
    if hardware.gpu_name:
        table.add_row("GPU", hardware.gpu_name)
    if hardware.npu_name:
        table.add_row("NPU", hardware.npu_name)
    if hardware.memory_bandwidth_gb_s:
        table.add_row("Est. peak bandwidth", f"~{hardware.memory_bandwidth_gb_s:.0f} GB/s")
    table.add_row("Fingerprint", hardware.fingerprint)
    console.print(table)
    console.print()


def _fmt(value: float | None, suffix: str = "", digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def print_benchmark_results(result: BenchmarkResult) -> None:
    console.print(
        Panel(
            f"[bold]{result.model_name}[/]  •  {result.backend}\n{result.hardware.summary}",
            border_style="green",
            expand=False,
        )
    )
    table = Table(title="Benchmark results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Notes")

    if result.speed:
        speed = result.speed
        table.add_row("Generation", _fmt(speed.tok_s_generation, " tok/s"), "measured")
        table.add_row("TTFT", _fmt(speed.ttft_seconds, " s", 2), "measured")
        if speed.sustained_tok_s is not None:
            table.add_row("Sustained", _fmt(speed.sustained_tok_s, " tok/s"), "measured")
        if speed.throttle_percent is not None:
            table.add_row("Throughput change", _fmt(speed.throttle_percent, "%"), "heuristic")
        if speed.thinking_tokens:
            table.add_row("Reasoning proxy", f"~{speed.thinking_tokens} tokens", "approximate")

    if result.memory:
        memory = result.memory
        table.add_row(
            "Peak system RAM",
            _fmt(memory.peak_system_ram_mb / 1024, " GB"),
            "sampled",
        )
        table.add_row(
            "Inference RAM delta",
            _fmt(memory.inference_delta_mb, " MB"),
            "system-wide",
        )
        table.add_row("RAM utilization", _fmt(memory.ram_utilization_pct, "%"), "system-wide")

    if result.power:
        power = result.power
        table.add_row(
            "Power",
            _fmt(power.avg_power_watts, " W"),
            f"{power.source}: {power.method}",
        )
        table.add_row("Efficiency", _fmt(power.tok_s_per_watt, " tok/s/W", 2), power.source)
        table.add_row("Energy/token", _fmt(power.energy_per_token_mj, " mJ", 2), power.source)

    if result.quality:
        quality = result.quality
        if quality.mmlu_accuracy is not None:
            table.add_row(
                "mini-MMLU",
                _fmt(quality.mmlu_accuracy, "%"),
                f"{quality.mmlu_total} questions",
            )
        if quality.humaneval_enabled:
            table.add_row(
                "mini-HumanEval",
                _fmt(quality.humaneval_pass_rate, "%"),
                "opt-in code execution",
            )

    console.print(table)
    console.print()
    if result.edge_score is not None:
        qualifier = "partial" if result.edge_score_is_partial else "full"
        console.print(
            Panel(
                f"[bold]Edge Score v{result.edge_score_version}: {result.edge_score}/100[/] "
                f"[dim]({qualifier})[/]",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()


def print_comparison(results: list[BenchmarkResult]) -> None:
    table = Table(title="Model comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    for result in results:
        table.add_column(result.model_name, justify="right")
    table.add_row(
        "Generation tok/s",
        *[_fmt(r.speed.tok_s_generation if r.speed else None) for r in results],
    )
    table.add_row(
        "TTFT",
        *[_fmt(r.speed.ttft_seconds if r.speed else None, " s", 2) for r in results],
    )
    table.add_row(
        "Peak system RAM",
        *[_fmt(r.memory.peak_system_ram_mb / 1024 if r.memory else None, " GB") for r in results],
    )
    table.add_row(
        "Edge Score",
        *[
            (f"{r.edge_score}/100*" if r.edge_score_is_partial else f"{r.edge_score}/100")
            if r.edge_score is not None
            else "—"
            for r in results
        ],
    )
    console.print(table)
    console.print("[dim]* partial score; compare only runs produced with the same protocol.[/]")
