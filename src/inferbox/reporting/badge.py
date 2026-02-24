"""Badge generation for InferBox results.

Creates shields.io badge URLs and SVG badges for README embedding.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from inferbox.models import BenchmarkResult


def _score_color(score: int) -> str:
    """Map Edge Score to badge color."""
    if score >= 80:
        return "brightgreen"
    elif score >= 60:
        return "green"
    elif score >= 40:
        return "yellow"
    elif score >= 20:
        return "orange"
    else:
        return "red"


def edge_score_badge_url(result: BenchmarkResult) -> str:
    """Generate shields.io badge URL for Edge Score."""
    score = result.edge_score or 0
    color = _score_color(score)
    label = quote("InferBox Edge Score")
    return f"https://img.shields.io/badge/{label}-{score}%2F100-{color}"


def tok_s_badge_url(result: BenchmarkResult) -> str:
    """Generate shields.io badge URL for tok/s."""
    if not result.speed:
        return ""
    tok_s = result.speed.tok_s_generation
    if tok_s >= 15:
        color = "brightgreen"
    elif tok_s >= 8:
        color = "green"
    elif tok_s >= 3:
        color = "yellow"
    else:
        color = "red"
    label = quote("tok/s")
    return f"https://img.shields.io/badge/{label}-{tok_s:.1f}-{color}"


def generate_badge_markdown(result: BenchmarkResult) -> str:
    """Generate full Markdown badge string for README embedding.

    Example output:
        ![InferBox Edge Score](https://img.shields.io/badge/...) ![tok/s](...)
    """
    parts = []

    # Edge Score badge
    score_url = edge_score_badge_url(result)
    parts.append(f"![InferBox Edge Score]({score_url})")

    # Tok/s badge
    if result.speed:
        toks_url = tok_s_badge_url(result)
        if toks_url:
            parts.append(f"![tok/s]({toks_url})")

    # Model badge
    model_name = quote(result.model_name)
    parts.append(
        f"![model](https://img.shields.io/badge/model-{model_name}-blue)"
    )

    return " ".join(parts)
