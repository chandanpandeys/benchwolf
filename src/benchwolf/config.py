"""BenchWolf configuration constants."""

from pathlib import Path

WARMUP_RUNS = 1
BENCHMARK_RUNS = 5
SUSTAINED_DURATION_SECONDS = 120
MAX_TOKENS_PER_RUN = 256
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05
POWER_SAMPLE_INTERVAL_SECONDS = 0.10

SPEED_PROMPT = (
    "Write a Python function named merge_sorted_lists that merges two sorted integer lists. "
    "Include type hints and a short docstring."
)
SUSTAINED_PROMPT = (
    "Implement a binary search tree in Python with insert, search, delete, traversal, height, "
    "and balance-check methods. Include type hints and concise complexity notes."
)
POWER_PROMPT = "Write an efficient Python function that sorts a list and explain its complexity."

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300

RATING_THRESHOLDS = {
    "tok_s": [(20, 5), (14, 4), (8, 3), (4, 2), (0, 1)],
    "ttft": [(0.5, 5), (1.0, 4), (2.0, 3), (4.0, 2), (float("inf"), 1)],
    "power_w": [(5, 5), (8, 4), (12, 3), (20, 2), (float("inf"), 1)],
    "tok_s_per_w": [(3.0, 5), (2.0, 4), (1.0, 3), (0.5, 2), (0, 1)],
    "throttle_pct": [(2, 5), (5, 4), (10, 3), (20, 2), (float("inf"), 1)],
    "mmlu_pct": [(70, 5), (60, 4), (50, 3), (40, 2), (0, 1)],
}

DATA_DIR = Path(__file__).parent / "data"
