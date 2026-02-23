"""TinyBench configuration constants."""

from pathlib import Path

# --- Benchmark Defaults ---
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5
SUSTAINED_DURATION_SECONDS = 120  # 2 minutes for sustained benchmark
MAX_TOKENS_PER_RUN = 256
THROTTLE_WINDOW_SECONDS = 30

# --- Default Prompts ---
SPEED_PROMPT = (
    "Write a Python function called `merge_sorted_lists` that takes two sorted lists "
    "of integers and returns a single sorted list containing all elements from both lists. "
    "Include docstring and type hints. Then write 3 unit tests for it."
)

SUSTAINED_PROMPT = (
    "You are a senior Python developer. Write a complete implementation of a binary search tree "
    "with the following methods: insert, delete, search, inorder_traversal, preorder_traversal, "
    "postorder_traversal, find_min, find_max, height, is_balanced, and level_order_traversal. "
    "Include comprehensive docstrings, type hints, and at least 10 unit tests using pytest. "
    "Add detailed comments explaining the time and space complexity of each method."
)

# --- Ollama Defaults ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300  # 5 minutes

# --- Rating Thresholds ---
RATING_THRESHOLDS = {
    "tok_s": [(20, 5), (14, 4), (8, 3), (4, 2), (0, 1)],
    "ttft": [(0.5, 5), (1.0, 4), (2.0, 3), (4.0, 2), (float("inf"), 1)],
    "ram_gb": [(2.0, 5), (3.0, 4), (4.0, 3), (6.0, 2), (float("inf"), 1)],
    "power_w": [(5, 5), (8, 4), (12, 3), (20, 2), (float("inf"), 1)],
    "tok_s_per_w": [(3.0, 5), (2.0, 4), (1.0, 3), (0.5, 2), (0, 1)],
    "throttle_pct": [(2, 5), (5, 4), (10, 3), (20, 2), (float("inf"), 1)],
    "mmlu_pct": [(70, 5), (60, 4), (50, 3), (40, 2), (0, 1)],
    "humaneval_pct": [(50, 5), (35, 4), (20, 3), (10, 2), (0, 1)],
}

# --- Paths ---
DATA_DIR = Path(__file__).parent / "data"
