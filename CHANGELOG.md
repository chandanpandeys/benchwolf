# Changelog

All notable changes to BenchWolf are documented here.

## [0.1.0] - Unreleased

### Added
- BenchWolf package and `benchwolf` CLI identity.
- Ollama and optional llama.cpp backends.
- Throughput, TTFT, sustained behavior, system-memory pressure, power, and quality benchmarks.
- Hardware/model preflight estimates and local result leaderboard.
- Edge Score v1 with explicit full/partial score semantics.
- Measurement provenance for power results.
- `SECURITY.md` and explicit opt-in mini-HumanEval execution.

### Changed
- Memory reporting now uses sampled system-wide RAM terminology rather than claiming model-only RAM.
- Power sampling runs concurrently with generation and no longer fabricates fallback values.
- Reasoning-token output is labeled approximate.
- CI separates lint, tests, and package validation.

### Security
- Model-generated Python is never executed during the default benchmark path.
