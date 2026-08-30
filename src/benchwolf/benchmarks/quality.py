"""Lightweight quality checks for BenchWolf."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.progress import Progress

from benchwolf.backends.base import Backend
from benchwolf.config import DATA_DIR
from benchwolf.models import QualityResult


def run_quality_benchmark(
    backend: Backend,
    progress: Progress | None = None,
    allow_code_execution: bool = False,
) -> QualityResult:
    """Run mini-MMLU and optionally mini-HumanEval."""
    mmlu_accuracy, mmlu_total = _run_mini_mmlu(backend, progress)
    humaneval_pass = humaneval_total = None
    if allow_code_execution:
        humaneval_pass, humaneval_total = _run_mini_humaneval(backend, progress)

    return QualityResult(
        mmlu_accuracy=mmlu_accuracy,
        mmlu_total=mmlu_total,
        humaneval_pass_rate=humaneval_pass,
        humaneval_total=humaneval_total,
        humaneval_enabled=allow_code_execution,
    )


def _run_mini_mmlu(
    backend: Backend,
    progress: Progress | None = None,
) -> tuple[float | None, int | None]:
    path = DATA_DIR / "mmlu_mini.json"
    if not path.exists():
        return None, None
    questions = json.loads(path.read_text(encoding="utf-8"))
    task = progress.add_task("[green]mini-MMLU...", total=len(questions)) if progress else None
    correct = 0
    for question in questions:
        result = backend.generate(_format_mmlu_prompt(question), max_tokens=5)
        answer = _extract_answer(result.text)
        if answer and answer == question["answer"].upper():
            correct += 1
        if progress and task is not None:
            progress.advance(task)
    total = len(questions)
    return (round(correct / total * 100.0, 1) if total else 0.0), total


def _format_mmlu_prompt(question: dict) -> str:
    choices = question["choices"]
    return (
        "Answer the multiple-choice question. Reply with ONLY A, B, C, or D.\n\n"
        f"Question: {question['question']}\n"
        f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nAnswer:"
    )


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text.strip()
    patterns = (
        r"(?:correct\s+)?answer\s+(?:is|would\s+be)\s*:?\s*[*(]*([A-Da-d])",
        r"answer\s*:\s*[*(]*([A-Da-d])",
        r"(?:option|choice)\s+[*(]*([A-Da-d])",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    first = cleaned.splitlines()[0].strip() if cleaned else ""
    match = re.match(r"^[*(]*([A-Da-d])(?:[*)\].:\s]|$)", first)
    if match:
        return match.group(1).upper()
    bracketed = re.search(r"[*(\[{]\s*([A-Da-d])\s*[*)\]}]", cleaned)
    if bracketed:
        return bracketed.group(1).upper()
    standalone = re.findall(r"\b([A-Da-d])\b", cleaned)
    return standalone[-1].upper() if standalone else None


def _run_mini_humaneval(
    backend: Backend,
    progress: Progress | None = None,
) -> tuple[float | None, int | None]:
    path = DATA_DIR / "humaneval_mini.json"
    if not path.exists():
        return None, None
    problems = json.loads(path.read_text(encoding="utf-8"))
    task = (
        progress.add_task("[blue]mini-HumanEval (code execution enabled)...", total=len(problems))
        if progress
        else None
    )
    passed = 0
    for problem in problems:
        completion = backend.generate(problem["prompt"], max_tokens=200)
        if _test_code(problem["prompt"], completion.text, problem["tests"]):
            passed += 1
        if progress and task is not None:
            progress.advance(task)
    total = len(problems)
    return (round(passed / total * 100.0, 1) if total else 0.0), total


def _test_code(prompt: str, completion: str, tests: str, timeout_s: int = 10) -> bool:
    """Execute generated code in an isolated-mode subprocess, not a security sandbox."""
    source = prompt + completion + "\n" + tests
    try:
        with tempfile.TemporaryDirectory(prefix="benchwolf-humaneval-") as temp_dir:
            script = Path(temp_dir) / "candidate.py"
            script.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
