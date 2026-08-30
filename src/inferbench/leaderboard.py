"""Leaderboard — save, view, and submit benchmark results.

Results are stored locally in ~/.inferbench/results/ as JSON files.
The `--submit` flag will upload to the public leaderboard API (when available).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from inferbench.models import BenchmarkResult

# Local storage directory
RESULTS_DIR = Path.home() / ".inferbench" / "results"


def _ensure_results_dir() -> Path:
    """Create the results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _generate_result_id(result: BenchmarkResult) -> str:
    """Generate a unique ID for a benchmark result."""
    key = f"{result.model_name}-{result.hardware.fingerprint}-{time.time()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def save_result(result: BenchmarkResult) -> Path:
    """Save a benchmark result to the local leaderboard.

    Returns:
        Path to the saved JSON file.
    """
    _ensure_results_dir()

    result_id = _generate_result_id(result)
    safe_model = result.model_name.replace(":", "_").replace("/", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_model}_{timestamp}_{result_id}.json"

    filepath = RESULTS_DIR / filename

    # Build leaderboard entry with extra metadata
    entry = {
        "id": result_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": json.loads(result.model_dump_json()),
        "meta": {
            "os": platform.system(),
            "python": platform.python_version(),
            "inferbench_version": result.inferbench_version,
        },
    }

    filepath.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return filepath


def load_all_results() -> list[dict]:
    """Load all saved benchmark results."""
    _ensure_results_dir()
    results = []

    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filepath"] = str(f)
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def submit_result(result: BenchmarkResult) -> dict:
    """Submit a benchmark result to the public leaderboard API.

    Returns:
        Response from the leaderboard API.
    """
    # For now, save locally and return a placeholder
    filepath = save_result(result)

    return {
        "status": "saved_locally",
        "filepath": str(filepath),
        "message": (
            "Result saved locally. Public leaderboard API coming soon!\n"
            "Results are stored in: ~/.inferbench/results/"
        ),
    }


def get_leaderboard_summary() -> list[dict]:
    """Get a summary of all saved results for display.

    Returns:
        List of summary dicts sorted by edge_score (descending).
    """
    all_results = load_all_results()
    summaries = []

    for entry in all_results:
        r = entry.get("result", {})
        hw = r.get("hardware", {})
        speed = r.get("speed", {})

        summaries.append({
            "model": r.get("model_name", "unknown"),
            "edge_score": r.get("edge_score", 0),
            "tok_s": speed.get("tok_s_generation", 0) if speed else 0,
            "cpu": hw.get("cpu_name", "unknown"),
            "ram_gb": hw.get("ram_total_gb", 0),
            "timestamp": entry.get("timestamp", ""),
            "backend": r.get("backend", "unknown"),
            "quantization": r.get("model_quantization", ""),
            "_filepath": entry.get("_filepath", ""),
        })

    # Sort by edge score descending
    summaries.sort(key=lambda x: x["edge_score"], reverse=True)
    return summaries
