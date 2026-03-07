# Wave 3 Performance Campaign

Date started: 2026-03-06
Branch: `perf/improvements-2`

## Scope
- Prioritize the diffusion-geometry / generic k-form pipeline.
- Keep the full existing test suite green.
- Keep one hypothesis per journal entry and per kept commit.

## Output layout
- `journal.md`: append-only hypothesis log.
- `baseline.md`: current Release baseline and hotspot ranking.
- `prior-art.md`: concise keep/avoid list from earlier campaigns.
- `results/`: benchmark JSON/TXT artifacts for this wave.
- `profiles/`: xctrace artifacts for this wave.

## Baseline commands
```bash
cmake --preset release-local
cmake --build build -j8
ctest --test-dir build --output-on-failure -j8
IGNEOUS_PERF_NOTES_DIR=notes/perf/20260306-wave3 IGNEOUS_BENCH_MODE=1 ./build/bench_geometry
IGNEOUS_PERF_NOTES_DIR=notes/perf/20260306-wave3 IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
IGNEOUS_PERF_NOTES_DIR=notes/perf/20260306-wave3 IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

## Acceptance policy
- Keep changes only when the primary target improves by at least `3%` on deep Release benchmarks.
- Reject changes that regress equal-or-higher-priority benchmarks by more than `3%`.
- Run full `ctest` before every kept commit.
