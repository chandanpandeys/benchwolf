"""Local result storage and leaderboard helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from benchwolf.models import BenchmarkResult

RESULTS_DIR = Path.home() / ".benchwolf" / "results"


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_result(result: BenchmarkResult) -> Path:
    _ensure_results_dir()
    key = f"{result.model_name}-{result.hardware.fingerprint}-{time.time()}"
    result_id = hashlib.sha256(key.encode()).hexdigest()[:12]
    model = result.model_name.replace(":", "_").replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{model}_{stamp}_{result_id}.json"
    entry = {
        "id": result_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result.model_dump(mode="json"),
        "meta": {
            "os": platform.system(),
            "python": platform.python_version(),
            "benchwolf_version": result.benchwolf_version,
        },
    }
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return path


def load_all_results() -> list[dict]:
    _ensure_results_dir()
    results: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["_filepath"] = str(path)
        results.append(data)
    return results


def get_leaderboard_summary() -> list[dict]:
    summaries = []
    for entry in load_all_results():
        result = entry.get("result", {})
        speed = result.get("speed") or {}
        hardware = result.get("hardware") or {}
        summaries.append(
            {
                "model": result.get("model_name", "unknown"),
                "edge_score": result.get("edge_score"),
                "partial": result.get("edge_score_is_partial", True),
                "tok_s": speed.get("tok_s_generation"),
                "cpu": hardware.get("cpu_name", "unknown"),
                "ram_gb": hardware.get("ram_total_gb", 0),
                "timestamp": entry.get("timestamp", ""),
                "backend": result.get("backend", "unknown"),
            }
        )
    summaries.sort(
        key=lambda item: item["edge_score"] if item["edge_score"] is not None else -1,
        reverse=True,
    )
    return summaries
