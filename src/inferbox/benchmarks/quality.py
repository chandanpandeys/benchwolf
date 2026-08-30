"""Quality benchmarks — mini-MMLU and mini-HumanEval."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from rich.progress import Progress

from inferbox.backends.base import Backend
from inferbox.config import DATA_DIR
from inferbox.models import QualityResult


def run_quality_benchmark(
    backend: Backend,
    progress: Optional[Progress] = None,
) -> QualityResult:
    """Run mini-MMLU and mini-HumanEval benchmarks."""
    mmlu_acc, mmlu_total = _run_mini_mmlu(backend, progress)
    humaneval_pass, humaneval_total = _run_mini_humaneval(backend, progress)

    return QualityResult(
        mmlu_accuracy=mmlu_acc,
        mmlu_total=mmlu_total,
        humaneval_pass_rate=humaneval_pass,
        humaneval_total=humaneval_total,
    )


def _run_mini_mmlu(
    backend: Backend,
    progress: Optional[Progress] = None,
) -> tuple[Optional[float], Optional[int]]:
    """Run mini-MMLU benchmark (100 multiple-choice questions)."""
    data_path = DATA_DIR / "mmlu_mini.json"
    if not data_path.exists():
        return None, None

    with open(data_path, encoding="utf-8") as f:
        questions = json.load(f)

    correct = 0
    total = len(questions)

    if progress:
        task = progress.add_task("[green]mini-MMLU...", total=total)

    for q in questions:
        prompt = _format_mmlu_prompt(q)
        result = backend.generate(prompt, max_tokens=5)
        answer = _extract_answer(result.text)

        if answer and answer.upper() == q["answer"].upper():
            correct += 1

        if progress:
            progress.advance(task)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return round(accuracy, 1), total


def _format_mmlu_prompt(question: dict) -> str:
    """Format a MMLU question as a prompt."""
    choices = question["choices"]
    prompt = (
        f"Answer the following multiple choice question. "
        f"Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Question: {question['question']}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"Answer:"
    )
    return prompt


def _extract_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D answer from model output using multi-strategy parsing.
    
    Handles:
    - Thinking tokens (<think>...</think>)
    - Explicit answers: 'The answer is D', 'Answer: C', 'Option B'
    - Formatted tokens: '**A**', '(B)', 'C.', 'D)'
    - Standalone single-letter responses
    """
    if not text:
        return None

    # Strip thinking blocks (e.g. DeepSeek R1, QwQ)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text.strip()

    # Strategy 1: Explicit answer phrases (case-insensitive)
    explicit_patterns = [
        r"(?:the\s+(?:correct\s+)?answer\s+(?:is|would\s+be)\s*:?\s*[\*\(]*([A-Da-d])[\*\)]*)",
        r"(?:answer\s*:\s*[\*\(]*([A-Da-d])[\*\)]*)",
        r"(?:option\s+[\*\(]*([A-Da-d])[\*\)]*)",
        r"(?:choice\s+[\*\(]*([A-Da-d])[\*\)]*)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Strategy 2: First line starting directly with option letter (e.g. "A", "(A)", "A.", "**A**", "A)")
    first_line = cleaned.split("\n")[0].strip()
    first_line_match = re.match(r"^[\*\(]*([A-Da-d])[\*\)\.\:\s]*", first_line)
    if first_line_match:
        # Avoid matching words like "All", "An", "Apple", "Because"
        candidate = first_line_match.group(1).upper()
        # Verify it's isolated or followed by punctuation/space
        if len(first_line) == 1 or not first_line[0:2].isalpha() or first_line.startswith(candidate + ")") or first_line.startswith(candidate + "."):
            return candidate

    # Strategy 3: Bolded or parenthesized letter anywhere in the text: (B), **B**, [B]
    bracket_match = re.search(r"[\*\({\[]\s*([A-Da-d])\s*[\*\)}\]]", cleaned)
    if bracket_match:
        return bracket_match.group(1).upper()

    # Strategy 4: Fallback - look for standalone single letter A-D token
    words = re.findall(r"\b([A-Da-d])\b", cleaned)
    if words:
        # Return the last standalone letter (often the concluding answer)
        return words[-1].upper()

    return None


def _run_mini_humaneval(
    backend: Backend,
    progress: Optional[Progress] = None,
) -> tuple[Optional[float], Optional[int]]:
    """Run mini-HumanEval benchmark (20 coding problems)."""
    data_path = DATA_DIR / "humaneval_mini.json"
    if not data_path.exists():
        return None, None

    with open(data_path, encoding="utf-8") as f:
        problems = json.load(f)

    passed = 0
    total = len(problems)

    if progress:
        task = progress.add_task("[blue]mini-HumanEval...", total=total)

    for problem in problems:
        prompt = problem["prompt"]

        # Timeout guard: skip if generation takes too long (small models are slow)
        result_container = [None]
        def _gen():
            result_container[0] = backend.generate(prompt, max_tokens=200)

        thread = threading.Thread(target=_gen)
        thread.start()
        thread.join(timeout=30)  # 30s max per problem

        if thread.is_alive() or result_container[0] is None:
            # Timed out — skip this problem
            if progress:
                progress.advance(task)
            continue

        if _test_code(problem["prompt"], result_container[0].text, problem["tests"]):
            passed += 1

        if progress:
            progress.advance(task)

    pass_rate = (passed / total) * 100 if total > 0 else 0.0
    return round(pass_rate, 1), total


def _test_code(prompt: str, completion: str, tests: str) -> bool:
    """Test generated code by executing it safely."""
    # Combine prompt + completion + tests
    full_code = prompt + completion + "\n" + tests

    try:
        # Run in subprocess for safety
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False
