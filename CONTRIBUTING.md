# Contributing to BenchWolf

Thanks for helping improve BenchWolf. Keep benchmark changes reproducible, cross-platform where practical, and explicit about whether a value is measured, estimated, or heuristic.

## Setup

```bash
git clone https://github.com/chandanpandeys/benchwolf.git
cd benchwolf
python -m pip install -e ".[dev]"
```

Before opening a pull request:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
python -m build
python -m twine check dist/*
```

## Benchmark changes

When changing methodology, add regression tests and update the README. Changes that alter Edge Score meaning must also change or document the score version; do not silently make old and new scores look comparable.

## Security

Never make generated-code execution implicit. HumanEval or similar execution-based evaluators must remain explicit opt-in and document their isolation limitations.
