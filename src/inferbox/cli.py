"""InferBox CLI — Edge AI Benchmark Tool.

Usage:
    inferbox run --model qwen2.5:3b
    inferbox info
    inferbox compare --models "qwen2.5:3b,phi3:3.8b"
    inferbox preflight
    inferbox leaderboard
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import click
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from inferbox import __version__
from inferbox.backends.base import Backend
from inferbox.backends.ollama import OllamaBackend
from inferbox.benchmarks.inference import run_speed_benchmark
from inferbox.benchmarks.memory import run_memory_benchmark
from inferbox.benchmarks.quality import run_quality_benchmark
from inferbox.config import RATING_THRESHOLDS
from inferbox.hardware.detect import detect_hardware
from inferbox.hardware.power import PowerMeter
from inferbox.models import BenchmarkResult, PowerResult
from inferbox.reporting.console import (
    console,
    print_benchmark_results,
    print_comparison,
    print_hardware_info,
    print_header,
)
from inferbox.reporting.export import export_json, export_markdown


def _get_backend(backend_name: str) -> Backend:
    """Get the appropriate backend."""
    if backend_name == "ollama":
        return OllamaBackend()
    elif backend_name == "llamacpp":
        try:
            from inferbox.backends.llamacpp import LlamaCppBackend
            return LlamaCppBackend()
        except ImportError:
            console.print(
                "[red]llama-cpp-python not installed.[/] "
                "Install with: pip install inferbox[llamacpp]"
            )
            sys.exit(1)
    else:
        console.print(f"[red]Unknown backend: {backend_name}[/]")
        sys.exit(1)


def _calculate_edge_score(result: BenchmarkResult) -> int:
    """Calculate composite Edge Score (0-100)."""
    scores = []
    weights = []

    if result.speed:
        # tok/s score (0-100)
        tok_s = result.speed.tok_s_generation
        tok_s_score = min(100, (tok_s / 20) * 100)  # 20 tok/s = 100
        scores.append(tok_s_score)
        weights.append(30)  # 30% weight

        # TTFT score
        ttft = result.speed.ttft_seconds
        ttft_score = max(0, min(100, (1 - (ttft - 0.3) / 3.7) * 100))
        scores.append(ttft_score)
        weights.append(10)

        # Throttle score
        if result.speed.throttle_percent is not None:
            throttle = abs(result.speed.throttle_percent)
            throttle_score = max(0, min(100, (1 - throttle / 20) * 100))
            scores.append(throttle_score)
            weights.append(5)

    if result.memory:
        # RAM efficiency score
        model_gb = result.memory.model_ram_mb / 1024
        ram_score = max(0, min(100, (1 - (model_gb - 1) / 7) * 100))
        scores.append(ram_score)
        weights.append(20)

    if result.power:
        # tok/s per watt score
        tpw = result.power.tok_s_per_watt
        power_score = min(100, (tpw / 3) * 100)  # 3 tok/s/W = 100
        scores.append(power_score)
        weights.append(15)

    if result.quality:
        if result.quality.mmlu_accuracy is not None:
            mmlu_score = result.quality.mmlu_accuracy  # Already 0-100
            scores.append(mmlu_score)
            weights.append(10)
        if result.quality.humaneval_pass_rate is not None:
            he_score = result.quality.humaneval_pass_rate  # Already 0-100
            scores.append(he_score)
            weights.append(10)

    if not scores:
        return 0

    # Weighted average
    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return round(weighted_sum / total_weight)


# ── CLI Commands ──────────────────────────────────────────────────────


@click.group()
@click.version_option(__version__, prog_name="inferbox")
def main():
    """🔬 InferBox — Edge AI Benchmark Tool.

    Measure LLM inference performance on any hardware in one command.
    """
    pass


@main.command()
@click.option("--model", "-m", required=True, help="Model name (e.g., qwen2.5:3b)")
@click.option(
    "--backend", "-b",
    type=click.Choice(["ollama", "llamacpp"]),
    default="ollama",
    help="Inference backend (default: ollama)",
)
@click.option(
    "--only",
    type=click.Choice(["speed", "memory", "power", "quality", "all"]),
    default="all",
    help="Run only specific benchmark category",
)
@click.option("--runs", "-r", default=5, help="Number of benchmark runs (default: 5)")
@click.option("--max-tokens", default=256, help="Max tokens per generation (default: 256)")
@click.option("--no-sustained", is_flag=True, help="Skip sustained/throttle test")
@click.option("--quick", is_flag=True, help="Run a quick 1-minute benchmark (fewer runs, no quality, no sustained)")
@click.option(
    "--export",
    type=click.Choice(["json", "markdown"]),
    default=None,
    help="Export format",
)
@click.option("--output", "-o", default=None, help="Output file path for export")
def run(
    model: str,
    backend: str,
    only: str,
    runs: int,
    max_tokens: int,
    no_sustained: bool,
    quick: bool,
    export: Optional[str],
    output: Optional[str],
):
    """Run benchmarks on a model."""
    print_header()

    # Detect hardware
    console.print("[dim]Detecting hardware...[/]")
    hw = detect_hardware()
    print_hardware_info(hw)

    # Initialize backend
    console.print(f"[dim]Connecting to {backend} backend...[/]")
    be = _get_backend(backend)

    if not be.is_available():
        console.print(f"[red]✗ {backend} is not available.[/]")
        if backend == "ollama":
            console.print("[yellow]  Start Ollama with: ollama serve[/]")
        sys.exit(1)

    console.print(f"[green]✓ {backend} is running[/]")

    # Load model
    console.print(f"[dim]Loading model {model}...[/]")
    try:
        be.load_model(model)
    except RuntimeError as e:
        console.print(f"[red]✗ {e}[/]")
        sys.exit(1)

    model_info = be.get_model_info()
    console.print(f"[green]✓ Model loaded: {model_info.get('parameter_size', 'unknown')}[/]")
    console.print()

    # Run benchmarks
    run_speed = only in ("all", "speed")
    run_memory = only in ("all", "memory")
    run_power = only in ("all", "power")
    run_quality = only in ("all", "quality")

    if quick:
        runs = min(runs, 2)
        no_sustained = True
        if only == "all":
            run_quality = False

    result = BenchmarkResult(
        inferbox_version=__version__,
        model_name=model,
        model_quantization=model_info.get("quantization"),
        backend=backend,
        hardware=hw,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        # Speed benchmark
        if run_speed:
            console.print("[bold cyan]🏎️  Running speed benchmarks...[/]")
            result.speed = run_speed_benchmark(
                be,
                runs=runs,
                max_tokens=max_tokens,
                include_sustained=not no_sustained,
                progress=progress,
            )

        # Memory benchmark
        if run_memory:
            console.print("[bold cyan]💾 Running memory benchmarks...[/]")
            result.memory = run_memory_benchmark(be, max_tokens=max_tokens)

        # Power benchmark
        if run_power:
            console.print("[bold cyan]⚡ Measuring power consumption...[/]")
            power_meter = PowerMeter()
            console.print(f"[dim]  Power method: {power_meter.method}[/]")

            power_meter.start()

            # Run a few generations while measuring power
            power_samples = 0
            for _ in range(3):
                be.generate(
                    "Write a Python function to sort a list efficiently.",
                    max_tokens=max_tokens,
                )
                power_meter.sample()
                power_samples += 1

            measurement = power_meter.stop()

            tok_s = result.speed.tok_s_generation if result.speed else 1.0
            avg_watts = measurement.avg_watts if measurement.avg_watts > 0 else 1.0

            result.power = PowerResult(
                method=measurement.method,
                avg_power_watts=round(avg_watts, 1),
                tok_s_per_watt=round(tok_s / avg_watts, 2),
                energy_per_token_mj=round((avg_watts / tok_s) * 1000, 1) if tok_s > 0 else 0.0,
                measurement_duration_s=round(measurement.duration_s, 1),
            )

        # Quality benchmark
        if run_quality:
            console.print("[bold cyan]🎯 Running quality benchmarks...[/]")
            result.quality = run_quality_benchmark(be, progress=progress)

    # Calculate edge score
    result.edge_score = _calculate_edge_score(result)

    # Print results
    console.print()
    print_benchmark_results(result)

    # Export if requested
    if export and output:
        if export == "json":
            export_json(result, output)
            console.print(f"[green]✓ Results exported to {output}[/]")
        elif export == "markdown":
            export_markdown(result, output)
            console.print(f"[green]✓ Results exported to {output}[/]")
    elif export and not output:
        # Default output filename
        safe_model = model.replace(":", "_").replace("/", "_")
        if export == "json":
            output = f"inferbox_{safe_model}.json"
            export_json(result, output)
        elif export == "markdown":
            output = f"inferbox_{safe_model}.md"
            export_markdown(result, output)
        console.print(f"[green]✓ Results exported to {output}[/]")

    # Auto-save to local leaderboard
    from inferbox.leaderboard import save_result
    saved_path = save_result(result)
    console.print(f"[dim]✓ Result saved to {saved_path}[/]")
    console.print("[dim]  View all results: inferbox leaderboard[/]")


@main.command()
def info():
    """Show detected hardware information."""
    print_header()
    hw = detect_hardware()
    print_hardware_info(hw)

    # Check backends
    console.print("[bold]Backend Status:[/]")

    ollama = OllamaBackend()
    if ollama.is_available():
        console.print("  [green]✓ Ollama is running[/]")
    else:
        console.print("  [red]✗ Ollama not found[/] — install from https://ollama.ai")

    try:
        import llama_cpp
        console.print(f"  [green]✓ llama-cpp-python v{llama_cpp.__version__}[/]")
    except ImportError:
        console.print("  [dim]○ llama-cpp-python not installed (optional)[/]")

    # Check power measurement
    power_meter = PowerMeter()
    console.print(f"\n[bold]Power Measurement:[/]")
    console.print(f"  Method: {power_meter.method}")


@main.command()
@click.option("--models", "-m", required=True, help="Comma-separated model names")
@click.option(
    "--backend", "-b",
    type=click.Choice(["ollama", "llamacpp"]),
    default="ollama",
)
@click.option("--runs", "-r", default=3, help="Runs per model (default: 3)")
@click.option("--max-tokens", default=256, help="Max tokens per generation")
def compare(models: str, backend: str, runs: int, max_tokens: int):
    """Compare multiple models side-by-side."""
    print_header()

    model_list = [m.strip() for m in models.split(",")]
    if len(model_list) < 2:
        console.print("[red]Need at least 2 models to compare. Separate with commas.[/]")
        sys.exit(1)

    console.print(f"[bold]Comparing {len(model_list)} models:[/]")
    for m in model_list:
        console.print(f"  • {m}")
    console.print()

    hw = detect_hardware()
    be = _get_backend(backend)

    if not be.is_available():
        console.print(f"[red]✗ {backend} is not available.[/]")
        sys.exit(1)

    results: list[BenchmarkResult] = []

    for model_name in model_list:
        console.print(f"\n[bold cyan]═══ Benchmarking: {model_name} ═══[/]")

        try:
            be.load_model(model_name)
        except RuntimeError as e:
            console.print(f"[red]✗ {e}[/]")
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            speed = run_speed_benchmark(
                be, runs=runs, max_tokens=max_tokens,
                include_sustained=False, progress=progress,
            )

        memory = run_memory_benchmark(be, max_tokens=max_tokens)

        model_info = be.get_model_info()
        result = BenchmarkResult(
            inferbox_version=__version__,
            model_name=model_name,
            model_quantization=model_info.get("quantization"),
            backend=backend,
            hardware=hw,
            speed=speed,
            memory=memory,
        )
        result.edge_score = _calculate_edge_score(result)
        results.append(result)

    if len(results) >= 2:
        console.print()
        print_comparison(results)


@main.command()
@click.option("--model", "-m", default=None, help="Check a specific model (optional)")
def preflight(model: Optional[str]):
    """Check if your hardware can run LLMs — no downloads needed."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from inferbox.preflight import FitStatus, run_preflight

    print_header()
    console.print("[bold]🔍 Preflight Hardware Check[/]")
    console.print("[dim]Checking what your hardware can run — no downloads needed.[/]")
    console.print()

    report = run_preflight(specific_model=model)

    # Hardware summary
    hw = report.hardware
    hw_table = Table(show_header=False, box=None, padding=(0, 2))
    hw_table.add_column(style="bold cyan")
    hw_table.add_column()
    hw_table.add_row("CPU", hw.cpu_name)
    hw_table.add_row("Cores", f"{hw.cpu_cores_physical}P / {hw.cpu_cores_logical}L ({hw.cpu_arch})")
    if hw.gpu_name:
        hw_table.add_row("GPU", hw.gpu_name)
    if hw.npu_name:
        hw_table.add_row("NPU / AI Engine", f"[bright_magenta]{hw.npu_name}[/]")
    hw_table.add_row("RAM Total", f"{hw.ram_total_gb:.1f} GB")
    hw_table.add_row("RAM Free Now", f"[dim]{report.available_ram_gb:.1f} GB[/]")
    hw_table.add_row("RAM Usable", f"[{'green' if report.usable_ram_gb > 3 else 'yellow' if report.usable_ram_gb > 1.5 else 'red'}]{report.usable_ram_gb:.1f} GB[/] [dim](total minus OS)[/]")
    if hw.memory_bandwidth_gb_s:
        hw_table.add_row("Est. Bandwidth", f"~{hw.memory_bandwidth_gb_s:.0f} GB/s [dim](memory bus)[/]")
    hw_table.add_row("Disk Free", f"{report.free_disk_gb:.1f} GB")
    hw_table.add_row("Ollama", "[green]Running ✓[/]" if report.ollama_running else "[dim]Not detected[/]")

    console.print(Panel(hw_table, title="[bold]💻 Your Hardware[/]", border_style="blue"))
    console.print()

    # Model compatibility table
    model_table = Table(title="[bold]📊 Model Compatibility[/]")
    model_table.add_column("Model", style="bold")
    model_table.add_column("Size", justify="right")
    model_table.add_column("RAM Needed", justify="right")
    model_table.add_column("Download", justify="right")
    model_table.add_column("Est. Speed", justify="right")
    model_table.add_column("Status")

    for r in report.model_results:
        # Color based on status
        if r.status == FitStatus.EASY:
            status_style = "green"
        elif r.status == FitStatus.GOOD:
            status_style = "green"
        elif r.status == FitStatus.TIGHT:
            status_style = "yellow"
        else:
            status_style = "red"

        tok_s_str = f"~{r.estimated_tok_s} t/s" if r.estimated_tok_s else "—"
        if r.bandwidth_ceiling_tok_s and r.status in (FitStatus.EASY, FitStatus.GOOD):
            tok_s_str = f"~{r.estimated_tok_s} t/s [dim](max {r.bandwidth_ceiling_tok_s:.0f})[/]"

        size_label = f"{r.model.params_b}B"
        if r.model.is_moe and r.model.active_params_b:
            size_label = f"{r.model.active_params_b}B/{r.model.params_b}B"

        model_table.add_row(
            r.model.name,
            size_label,
            f"{r.model.min_ram_gb:.1f} GB",
            f"{r.model.disk_size_gb:.1f} GB",
            tok_s_str,
            f"[{status_style}]{r.status_emoji} {r.status_label}[/]",
        )

    console.print(model_table)
    console.print()

    # Recommendation
    if report.recommended_model:
        rec = report.recommended_model
        console.print(
            Panel(
                f"[bold green]Recommended:[/] Start with [bold]{rec}[/]\n"
                f"Best balance of quality and speed for your hardware.\n\n"
                f"[dim]Next steps:[/]\n"
                f"  1. Install Ollama: [cyan]https://ollama.ai[/]\n"
                f"  2. [bold]ollama pull {rec}[/]\n"
                f"  3. [bold]inferbox run --model {rec}[/]",
                title="[bold]💡 Recommendation[/]",
                border_style="green",
            )
        )
    elif not report.can_run_any:
        console.print(
            Panel(
                "[bold red]Your hardware cannot comfortably run any LLM models.[/]\n\n"
                "[dim]Minimum requirements for the smallest model (qwen2.5:0.5b):[/]\n"
                "  • 2 GB available RAM\n"
                "  • 2 GB free disk space\n\n"
                "[dim]Consider:[/]\n"
                "  • Closing other applications to free RAM\n"
                "  • Using a cloud-based LLM service instead\n"
                "  • Upgrading your hardware",
                title="[bold]⚠️  Hardware Insufficient[/]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]Some models may fit, but with tight margins.[/]\n"
                "Close other applications before benchmarking to free RAM.",
                title="[bold]⚠️  Limited Compatibility[/]",
                border_style="yellow",
            )
        )


@main.command()
@click.option("--clear", is_flag=True, help="Clear all saved results")
def leaderboard(clear: bool):
    """View your local benchmark leaderboard."""
    from rich.panel import Panel
    from rich.table import Table

    from inferbox.leaderboard import get_leaderboard_summary, RESULTS_DIR

    print_header()

    if clear:
        import shutil
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
            console.print("[green]✓ Leaderboard cleared.[/]")
        else:
            console.print("[dim]No results to clear.[/]")
        return

    summaries = get_leaderboard_summary()

    if not summaries:
        console.print(
            Panel(
                "[dim]No benchmark results saved yet.[/]\n\n"
                "Run a benchmark first:\n"
                "  [bold]inferbox run --model qwen2.5:3b[/]",
                title="[bold]🏆 Leaderboard[/]",
                border_style="blue",
            )
        )
        return

    table = Table(title="[bold]🏆 Your Benchmark Leaderboard[/]")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Edge Score", justify="center")
    table.add_column("tok/s", justify="right")
    table.add_column("Backend", style="dim")
    table.add_column("CPU")
    table.add_column("RAM", justify="right")
    table.add_column("Date", style="dim")

    for i, s in enumerate(summaries, 1):
        score = s["edge_score"]
        if score >= 70:
            score_style = "bold green"
        elif score >= 40:
            score_style = "bold yellow"
        else:
            score_style = "bold red"

        # Format date
        date_str = s["timestamp"][:10] if s["timestamp"] else "—"

        table.add_row(
            str(i),
            s["model"],
            f"[{score_style}]{score}[/]",
            f"{s['tok_s']:.1f}" if s["tok_s"] else "—",
            s["backend"],
            s["cpu"][:30] + "..." if len(s["cpu"]) > 30 else s["cpu"],
            f"{s['ram_gb']:.0f} GB",
            date_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(summaries)} result(s) saved in {RESULTS_DIR}[/]")


if __name__ == "__main__":
    main()
