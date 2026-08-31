"""Shields.io badge helpers for BenchWolf results."""

from urllib.parse import quote

from benchwolf.models import BenchmarkResult


def _score_color(score: int) -> str:
    if score >= 80:
        return "brightgreen"
    if score >= 60:
        return "green"
    if score >= 40:
        return "yellow"
    if score >= 20:
        return "orange"
    return "red"


def edge_score_badge_url(result: BenchmarkResult) -> str:
    if result.edge_score is None:
        return ""
    label = "BenchWolf Edge Score"
    if result.edge_score_is_partial:
        label += " partial"
    return f"https://img.shields.io/badge/{quote(label)}-{result.edge_score}%2F100-{_score_color(result.edge_score)}"


def tok_s_badge_url(result: BenchmarkResult) -> str:
    if not result.speed:
        return ""
    value = result.speed.tok_s_generation
    color = "brightgreen" if value >= 15 else "green" if value >= 8 else "yellow" if value >= 3 else "red"
    return f"https://img.shields.io/badge/{quote('tok/s')}-{value:.1f}-{color}"


def generate_badge_markdown(result: BenchmarkResult) -> str:
    parts: list[str] = []
    score = edge_score_badge_url(result)
    if score:
        parts.append(f"![BenchWolf Edge Score]({score})")
    speed = tok_s_badge_url(result)
    if speed:
        parts.append(f"![tok/s]({speed})")
    return " ".join(parts)
