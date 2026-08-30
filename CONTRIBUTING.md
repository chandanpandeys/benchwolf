# Contributing to InferBox

Thank you for considering contributing to InferBox! Every contribution — from bug reports to code to documentation — helps make edge AI benchmarking better for everyone.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/chandanpandeys/inferbox.git
cd inferbox

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/
```

## How to Contribute

### 🐛 Bug Reports

Open an issue with:
- Your OS and Python version
- InferBox version (`inferbox --version`)
- Steps to reproduce
- Expected vs actual behavior
- Output of `inferbox info`

### 💡 Feature Requests

Open an issue and describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### 🔧 Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run `pytest tests/ -v` and `ruff check src/` to verify
5. Write a clear PR description explaining what and why
6. Submit the PR

### Good First Issues

Look for issues labeled `good first issue` — these are beginner-friendly tasks like:
- Adding new models to the model database
- Improving error messages
- Writing additional tests
- Documentation improvements

## Code Style

- **Python 3.10+** — use modern type hints
- **Ruff** for linting and formatting (`ruff check`, `ruff format`)
- **Line length:** 100 characters max
- **Docstrings:** Google-style for all public functions
- **Tests:** pytest, placed in `tests/`

## Architecture Overview

```
src/inferbox/
├── backends/       # Inference backend abstraction (Ollama, llama.cpp)
├── benchmarks/     # Benchmark implementations (speed, memory, quality)
├── data/           # Bundled evaluation data (mini-MMLU, mini-HumanEval)
├── hardware/       # Hardware detection and power measurement
├── reporting/      # Console output and export (JSON, Markdown)
├── cli.py          # CLI entry point (Click)
├── config.py       # Configuration constants
├── models.py       # Pydantic data models
├── preflight.py    # Hardware feasibility checker
└── leaderboard.py  # Local results storage
```

### Adding a New Backend

1. Create `src/inferbox/backends/your_backend.py`
2. Implement the `Backend` abstract class from `base.py`
3. Register it in `cli.py:_get_backend()`

### Adding a New Benchmark

1. Create `src/inferbox/benchmarks/your_benchmark.py`
2. Return a Pydantic model with your results
3. Wire it into the `run` command in `cli.py`

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
