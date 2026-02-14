# Hodge Parity Journal

Use one entry per parity hypothesis.

## Entry Template
- Timestamp:
- Commit:
- Hypothesis:
- Files touched:
- Commands:
- Numeric parity deltas:
- Decision: `kept` | `rejected` | `deferred`
- Notes:

## 2026-02-14 Baseline Harness + Gauge-Invariant Metrics
- Timestamp: 2026-02-14T16:09:21Z
- Commit: working tree (uncommitted)
- Hypothesis: Establish deterministic cross-language parity artifacts first, then compare harmonic forms in ambient space (not raw coefficient coordinates) to avoid basis-order artifacts.
- Files touched:
  - `include/igneous/data/topology.hpp`
  - `include/igneous/ops/geometry.hpp`
  - `include/igneous/ops/spectral.hpp`
  - `include/igneous/ops/hodge.hpp`
  - `src/main_hodge.cpp`
  - `src/main_diffusion.cpp`
  - `src/main_spectral.cpp`
  - `tests/test_topology_diffusion.cpp`
  - `tests/test_ops_spectral_geometry.cpp`
  - `tests/test_ops_hodge.cpp`
  - `benches/bench_dod.cpp`
  - `benches/bench_pipelines.cpp`
  - `scripts/hodge/setup_reference_env.sh`
  - `scripts/hodge/generate_input_torus.py`
  - `scripts/hodge/run_reference_hodge.py`
  - `scripts/hodge/run_cpp_hodge.sh`
  - `scripts/hodge/compare_hodge_outputs.py`
  - `scripts/hodge/run_parity_round.sh`
  - `notes/hodge/journal.md`
- Commands:
  - `cmake --build build -j8`
  - `ctest --test-dir build --output-on-failure`
  - `scripts/hodge/run_parity_round.sh`
  - `python3 scripts/hodge/compare_hodge_outputs.py --reference-dir notes/hodge/results/round_20260214-160921/reference --cpp-dir notes/hodge/results/round_20260214-160921/cpp --output-markdown notes/hodge/results/round_20260214-160921/report/parity_report.md --output-json notes/hodge/results/round_20260214-160921/report/parity_report.json --label \"Hodge parity round 20260214-160921\"`
- Numeric parity deltas:
  - Baseline report: `notes/hodge/results/round_20260214-160921/report/parity_report.json`
  - Composite score: `0.310740`
  - Harmonic subspace max principal angle: `9.229825 deg`
  - Harmonic Procrustes error: `0.129113`
  - Circular correlation min: `0.098627`
  - Circular wrapped P95 max: `2.936349 rad`
- Decision: `kept`
- Notes:
  - Concrete structural mismatch identified: coefficient-space harmonic comparison is not gauge-invariant due basis ordering differences.
  - Primary harmonic parity metrics now use `harmonic_ambient.csv` / `reference_harmonic_ambient.csv`.

## 2026-02-14 Hypothesis A1: Normalized Symmetric-Kernel Eigensolve
- Timestamp: 2026-02-14T16:11:23Z
- Commit: working tree (uncommitted)
- Hypothesis: Solving eigenvectors on `D^{-1/2} K D^{-1/2}` instead of `K` will improve harmonic-form parity.
- Files touched:
  - `include/igneous/ops/spectral.hpp` (reverted)
- Commands:
  - `cmake --build build -j8`
  - `PREVIOUS_REPORT_JSON=notes/hodge/results/round_20260214-160921/report/parity_report.json scripts/hodge/run_parity_round.sh`
  - `python3 scripts/hodge/compare_hodge_outputs.py --reference-dir notes/hodge/results/round_20260214-161123/reference --cpp-dir notes/hodge/results/round_20260214-161123/cpp --output-markdown notes/hodge/results/round_20260214-161123/report/parity_report.md --output-json notes/hodge/results/round_20260214-161123/report/parity_report.json --label \"Hodge parity round 20260214-161123\" --previous-json notes/hodge/results/round_20260214-160921/report/parity_report.json`
- Numeric parity deltas:
  - Report: `notes/hodge/results/round_20260214-161123/report/parity_report.json`
  - Composite: `0.249454` (`-19.72%` vs baseline)
  - Harmonic subspace max principal angle: `4.066515 deg` (improved)
  - Harmonic Procrustes error: `0.059640` (improved)
  - Circular correlation min: `0.050558` (regressed)
  - Circular wrapped P95 max: `2.964019 rad` (regressed)
- Decision: `rejected`
- Notes:
  - Improvement was below the required `20%` keep threshold.
  - Change discarded per commit gate policy.

## 2026-02-14 Hypothesis D1: Circular Mode Selection Parity
- Timestamp: 2026-02-14T16:18:50Z
- Commit: working tree (uncommitted)
- Hypothesis: The reference workflow uses the first positive-imaginary circular mode for each harmonic form; setting second mode default to `0` (not `1`) should restore circular parity.
- Files touched:
  - `src/main_hodge.cpp`
  - `scripts/hodge/run_reference_hodge.py`
  - `scripts/hodge/run_cpp_hodge.sh`
  - `scripts/hodge/run_parity_round.sh`
- Commands:
  - `cmake --build build -j8`
  - `PREVIOUS_REPORT_JSON=notes/hodge/results/round_20260214-160921/report/parity_report.json scripts/hodge/run_parity_round.sh`
- Numeric parity deltas:
  - Report: `notes/hodge/results/round_20260214-161850/report/parity_report.json`
  - Composite: `0.045839` (`-85.25%` vs baseline)
  - Harmonic subspace max principal angle: `4.155913 deg`
  - Harmonic Procrustes error: `0.060369`
  - Circular correlation min: `0.995021` (major improvement)
  - Circular wrapped P95 max: `0.170368 rad` (major improvement)
- Decision: `kept`
- Notes:
  - This is a structural parity fix in eigenmode selection policy, not a numerical micro-optimization.
  - Remaining gaps are now concentrated in harmonic-form thresholds and circular P95 tail.

## 2026-02-14 Hypothesis H1: Deterministic Reference Basis in Harness
- Timestamp: 2026-02-14T16:26:56Z
- Commit: `388e076` + working tree (uncommitted)
- Hypothesis: Reference parity runs are unstable because the reference eigensolver initialisation is random; precomputing a deterministic function basis from the same reference kernel path should stabilise comparisons.
- Files touched:
  - `scripts/hodge/run_reference_hodge.py`
- Commands:
  - `scripts/hodge/run_parity_round.sh` (repeated)
  - Determinism check:
    - repeated `scripts/hodge/run_reference_hodge.py` runs and checksum comparison on `reference_harmonic_coeffs.csv`
- Numeric parity deltas:
  - Deterministic baseline report: `notes/hodge/results/round_20260214-162656/report/parity_report.json`
  - Composite score: `0.156485`
  - Harmonic subspace max principal angle: `6.493845 deg`
  - Harmonic Procrustes error: `0.098484`
  - Circular correlation min: `0.714689`
  - Circular wrapped P95 max: `1.572800 rad`
- Decision: `kept`
- Notes:
  - Reference outputs are now repeatable across runs for identical inputs/parameters.
  - This intentionally re-baselines parity metrics for reliable hypothesis gating.

## 2026-02-14 Hypothesis A2: Normalized Symmetric-Kernel Eigensolve (Deterministic Baseline)
- Timestamp: 2026-02-14T16:28:53Z
- Commit: `388e076` + working tree (uncommitted)
- Hypothesis: Under deterministic reference outputs, solving the normalized symmetric operator `D^{-1/2} K D^{-1/2}` should materially reduce harmonic mismatch.
- Files touched:
  - `include/igneous/ops/spectral.hpp`
- Commands:
  - `cmake --build build -j8`
  - `PREVIOUS_REPORT_JSON=notes/hodge/results/round_20260214-162656/report/parity_report.json scripts/hodge/run_parity_round.sh`
  - Stability check: repeated `scripts/hodge/run_parity_round.sh` runs
- Numeric parity deltas:
  - Report: `notes/hodge/results/round_20260214-162853/report/parity_report.json`
  - Composite: `0.066904` (`-57.25%` vs deterministic baseline)
  - Harmonic subspace max principal angle: `6.376378 deg`
  - Harmonic Procrustes error: `0.097143`
  - Circular correlation min: `0.996571` (major improvement)
  - Circular wrapped P95 max: `0.163400 rad` (major improvement)
- Decision: `kept`
- Notes:
  - Cleared the 20% keep threshold with deterministic artifacts.
  - Harmonic thresholds remain the dominant blocker.

## 2026-02-14 Hypothesis H2: Double-Precision Hodge/Circular Solves
- Timestamp: 2026-02-14T16:30:47Z
- Commit: `388e076` + working tree (uncommitted)
- Hypothesis: Promoting Hodge spectral and circular generalized eigensolves to double precision will reduce harmonic/circular error tails.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Commands:
  - `cmake --build build -j8`
  - `PREVIOUS_REPORT_JSON=notes/hodge/results/round_20260214-162853/report/parity_report.json scripts/hodge/run_parity_round.sh`
- Numeric parity deltas:
  - Report: `notes/hodge/results/round_20260214-163047/report/parity_report.json`
  - Composite: `0.159232` (`+138.00%` regression)
  - Circular correlation min dropped to `0.714585`
  - Circular wrapped P95 max regressed to `1.566015 rad`
- Decision: `rejected`
- Notes:
  - Severe regression; reverted immediately.
