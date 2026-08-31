"""BenchWolf command-line interface."""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from benchwolf import __version__
from benchwolf.backends.base import Backend
from benchwolf.backends.ollama import OllamaBackend
from benchwolf.benchmarks.inference import run_speed_benchmark
from benchwolf.benchmarks.memory import run_memory_benchmark
from benchwolf.benchmarks.power import run_power_benchmark
from benchwolf.benchmarks.quality import run_quality_benchmark
from benchwolf.hardware.detect import detect_hardware
from benchwolf.hardware.power import PowerMeter
from benchwolf.leaderboard import RESULTS_DIR, get_leaderboard_summary, save_result
from benchwolf.models import BenchmarkResult
from benchwolf.reporting.console import (
    console,
    print_benchmark_results,
    print_comparison,
    print_hardware_info,
    print_header,
)
from benchwolf.reporting.export import export_json, export_markdown
from benchwolf.scoring import calculate_edge_score


def _get_backend(backend_name: str) -> Backend:
    if backend_name == "ollama":
        return OllamaBackend()
    if backend_name == "llamacpp":
        try:
            from benchwolf.backends.llamacpp import LlamaCppBackend
        except ImportError:
            console.print("[red]llama-cpp-python is not installed.[/] Install it with: pip install benchwolf[llamacpp]")
            raise click.Abort from None
        return LlamaCppBackend()
    raise click.ClickException(f"Unknown backend: {backend_name}")


def _apply_score(result: BenchmarkResult) -> None:
    summary = calculate_edge_score(result)
    result.edge_score = summary.score
    result.edge_score_is_partial = summary.partial


@click.group()
@click.version_option(__version__, prog_name="benchwolf")
def main() -> None:
    """Benchmark local LLMs on the hardware you actually own."""


@main.command()
@click.option("--model", "-m", required=True, help="Model name or GGUF path.")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp"]),
    default="ollama",
    show_default=True,
)
@click.option(
    "--only",
    type=click.Choice(["speed", "memory", "power", "quality", "all"]),
    default="all",
    show_default=True,
    help="Run one benchmark category or the full protocol.",
)
@click.option("--runs", "-r", default=5, type=click.IntRange(1, 50), show_default=True)
@click.option("--max-tokens", default=256, type=click.IntRange(1, 8192), show_default=True)
@click.option("--no-sustained", is_flag=True, help="Skip the sustained throughput check.")
@click.option("--quick", is_flag=True, help="Fewer runs; skips quality and sustained checks.")
@click.option(
    "--allow-code-execution",
    is_flag=True,
    help=("Opt in to mini-HumanEval. This executes model-generated Python locally and is NOT a security sandbox."),
)
@click.option("--export", "export_format", type=click.Choice(["json", "markdown"]))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path))
def run(
    model: str,
    backend: str,
    only: str,
    runs: int,
    max_tokens: int,
    no_sustained: bool,
    quick: bool,
    allow_code_execution: bool,
    export_format: str | None,
    output: Path | None,
) -> None:
    """Run BenchWolf against one model."""
    print_header()
    console.print("[dim]Detecting hardware...[/]")
    hardware = detect_hardware()
    print_hardware_info(hardware)

    adapter = _get_backend(backend)
    if not adapter.is_available():
        raise click.ClickException(f"{backend} is not available")

    console.print(f"[dim]Loading {model} with {backend}...[/]")
    try:
        adapter.load_model(model)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    model_info = adapter.get_model_info()
    result = BenchmarkResult(
        benchwolf_version=__version__,
        model_name=model,
        model_quantization=model_info.get("quantization"),
        backend=backend,
        hardware=hardware,
    )

    run_speed = only in {"all", "speed"}
    run_memory = only in {"all", "memory"}
    run_power = only in {"all", "power"}
    run_quality = only in {"all", "quality"}
    if quick:
        runs = min(runs, 2)
        no_sustained = True
        if only == "all":
            run_quality = False

    if allow_code_execution and not run_quality:
        console.print("[yellow]--allow-code-execution has no effect because quality benchmarks are skipped.[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        if run_speed:
            console.print("[bold cyan]🏎  Speed[/]")
            result.speed = run_speed_benchmark(
                adapter,
                runs=runs,
                max_tokens=max_tokens,
                include_sustained=not no_sustained,
                progress=progress,
            )
        if run_memory:
            console.print("[bold cyan]💾  System memory pressure[/]")
            result.memory = run_memory_benchmark(adapter, max_tokens=max_tokens)
        if run_power:
            console.print("[bold cyan]⚡  Power[/]")
            result.power = run_power_benchmark(adapter, max_tokens=max_tokens)
        if run_quality:
            if allow_code_execution:
                console.print("[bold yellow]⚠ mini-HumanEval code execution explicitly enabled.[/]")
            result.quality = run_quality_benchmark(
                adapter,
                progress=progress,
                allow_code_execution=allow_code_execution,
            )

    _apply_score(result)
    print_benchmark_results(result)

    if export_format:
        safe_model = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        destination = output or Path(f"benchwolf_{safe_model}.{'json' if export_format == 'json' else 'md'}")
        if export_format == "json":
            export_json(result, str(destination))
        else:
            export_markdown(result, str(destination))
        console.print(f"[green]✓ Exported to {destination}[/]")

    saved = save_result(result)
    console.print(f"[dim]Saved locally: {saved}[/]")
    console.print("[dim]View results with: benchwolf leaderboard[/]")


@main.command()
def info() -> None:
    """Show hardware detection and backend status."""
    print_header()
    hardware = detect_hardware()
    print_hardware_info(hardware)
    table = Table(title="Runtime status")
    table.add_column("Component")
    table.add_column("Status")
    table.add_row("Ollama", "available" if OllamaBackend().is_available() else "not detected")
    meter = PowerMeter()
    table.add_row("Power source", f"{meter.source}: {meter.method}")
    console.print(table)


@main.command()
@click.option("--models", "-m", required=True, help="Comma-separated model names.")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp"]),
    default="ollama",
    show_default=True,
)
@click.option("--runs", "-r", default=3, type=click.IntRange(1, 50), show_default=True)
@click.option("--max-tokens", default=256, type=click.IntRange(1, 8192), show_default=True)
def compare(models: str, backend: str, runs: int, max_tokens: int) -> None:
    """Compare speed and memory pressure for multiple models."""
    print_header()
    model_names = [item.strip() for item in models.split(",") if item.strip()]
    if len(model_names) < 2:
        raise click.ClickException("Provide at least two comma-separated models")

    hardware = detect_hardware()
    adapter = _get_backend(backend)
    if not adapter.is_available():
        raise click.ClickException(f"{backend} is not available")

    results: list[BenchmarkResult] = []
    for model_name in model_names:
        console.print(f"[bold cyan]Benchmarking {model_name}[/]")
        try:
            adapter.load_model(model_name)
        except RuntimeError as exc:
            console.print(f"[red]Skipping {model_name}: {exc}[/]")
            continue
        speed = run_speed_benchmark(
            adapter,
            runs=runs,
            max_tokens=max_tokens,
            include_sustained=False,
        )
        memory = run_memory_benchmark(adapter, max_tokens=max_tokens)
        result = BenchmarkResult(
            benchwolf_version=__version__,
            model_name=model_name,
            model_quantization=adapter.get_model_info().get("quantization"),
            backend=backend,
            hardware=hardware,
            speed=speed,
            memory=memory,
        )
        _apply_score(result)
        results.append(result)

    if len(results) < 2:
        raise click.ClickException("Fewer than two models completed successfully")
    print_comparison(results)


@main.command()
@click.option("--model", "-m", help="Check one known Ollama model instead of the full catalog.")
def preflight(model: str | None) -> None:
    """Estimate model fit and throughput without downloading model files."""
    from benchwolf.preflight import FitStatus, run_preflight

    print_header()
    report = run_preflight(specific_model=model)
    print_hardware_info(report.hardware)
    console.print(
        "[dim]Preflight speed values are heuristics; bandwidth ceilings are theoretical, "
        "not measured inference results.[/]"
    )

    table = Table(title="Model compatibility")
    table.add_column("Model", style="bold")
    table.add_column("RAM need", justify="right")
    table.add_column("Download", justify="right")
    table.add_column("Est. speed", justify="right")
    table.add_column("Status")
    for item in report.model_results:
        speed = "—" if item.estimated_tok_s is None else f"~{item.estimated_tok_s:.1f} tok/s"
        if item.bandwidth_ceiling_tok_s is not None:
            speed += f" (ceiling {item.bandwidth_ceiling_tok_s:.0f})"
        style = (
            "green"
            if item.status in {FitStatus.EASY, FitStatus.GOOD}
            else "yellow"
            if item.status == FitStatus.TIGHT
            else "red"
        )
        table.add_row(
            item.model.name,
            f"{item.model.min_ram_gb:.1f} GB",
            f"{item.model.disk_size_gb:.1f} GB",
            speed,
            f"[{style}]{item.status_emoji} {item.status_label}[/]",
        )
    console.print(table)
    if report.recommended_model:
        console.print(
            Panel(
                f"Try [bold]{report.recommended_model}[/] first.\n"
                f"ollama pull {report.recommended_model}\n"
                f"benchwolf run --model {report.recommended_model}",
                title="Recommendation",
                border_style="green",
            )
        )


@main.command()
@click.option("--clear", is_flag=True, help="Delete locally saved BenchWolf results.")
def leaderboard(clear: bool) -> None:
    """View locally saved benchmark results."""
    print_header()
    if clear:
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
            console.print("[green]✓ Local results cleared.[/]")
        return

    summaries = get_leaderboard_summary()
    if not summaries:
        console.print("No saved results. Run: benchwolf run --model qwen2.5:3b")
        return

    table = Table(title="Local leaderboard")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Edge Score", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Backend")
    table.add_column("Date")
    for index, item in enumerate(summaries, 1):
        score = item["edge_score"]
        score_text = "—" if score is None else f"{score}{'*' if item['partial'] else ''}"
        tok_s = item["tok_s"]
        table.add_row(
            str(index),
            item["model"],
            score_text,
            "—" if tok_s is None else f"{tok_s:.1f}",
            item["backend"],
            item["timestamp"][:10],
        )
    console.print(table)
    console.print(f"[dim]{RESULTS_DIR} • * partial score[/]")


if __name__ == "__main__":
    main()
