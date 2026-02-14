# Diffusion Geometry Journal

Use one entry per hypothesis or implementation batch.

## Entry Template
- Timestamp:
- Commit:
- Hypothesis:
- Files touched:
- Commands:
- Baseline:
- Results:
- Numeric checks:
- Decision: `kept` | `rejected` | `deferred`
- Notes:

## 2026-02-14 Wave 1 Baseline
- Timestamp: 2026-02-14T00:00:00Z
- Commit: 78ae77b (working tree before Wave 1 implementation)
- Hypothesis: Baseline current Hodge/spectral behavior and lock reproducible regression evidence before changes.
- Files touched:
  - `notes/diffusion/journal.md` (new)
- Commands:
  - `cmake --build build -j8`
  - `./build/igneous-hodge`
  - `ctest --test-dir build --output-on-failure`
- Baseline:
  - `igneous-hodge` large torus run (`n=4000`, `basis=64`) reports:
    - `lambda_0 = 0.0106422`
    - `lambda_1 = 0.0307465`
    - `lambda_2 = 0.056302`
  - This is inconsistent with expected near-harmonic low modes (`~1e-5` scale).
- Results:
  - Full existing test suite passes (`7/7`), indicating current tests do not detect this regression.
- Numeric checks:
  - `ctest`: pass
  - Hodge run: deterministic reproduction of non-harmonic low-mode issue
- Decision: `kept`
- Notes:
  - This entry establishes the pre-fix reference point for Wave 1.

## Math Regression Checklist
- [ ] `compute_eigenbasis` solver-path diagnostics and residual checks.
- [ ] Hodge low-mode envelope guard on canonical torus case.
- [ ] Circular-coordinate dynamic-range guard.
- [ ] 1-form decomposition reconstruction and orthogonality checks.
- [ ] `carre_du_champ` bilinearity/symmetry/positivity checks.
- [ ] transform/flow/curvature invariants and no-op checks.

## 2026-02-14 Spectral Solver Safety Gate
- Timestamp: 2026-02-14T00:00:00Z
- Commit: working tree after baseline
- Hypothesis: Restrict symmetric spectral solve to diffusion graphs that pass strict reversibility diagnostics; otherwise force generic Arnoldi path to avoid non-harmonic low Hodge modes.
- Files touched:
  - `include/igneous/data/topology.hpp`
  - `include/igneous/ops/spectral.hpp`
- Commands:
  - `cmake --build build -j8`
  - `./build/igneous-hodge`
  - `ctest --test-dir build --output-on-failure`
- Baseline:
  - Large torus run emitted `lambda_0 = 0.0106422`, `lambda_1 = 0.0307465`.
- Results:
  - Added additive spectral options (`auto|generic|symmetric`) and diagnostics.
  - Added topology-side spectral diagnostics and cached spectral eigenvalues.
  - Auto mode now reports reversibility and selects generic solve on non-reversible graph.
  - Large torus run now emits:
    - `lambda_0 = 2.56507e-05`
    - `lambda_1 = 3.29904e-05`
- Numeric checks:
  - `ctest`: pass (`7/7`)
- Decision: `kept`
- Notes:
  - Fix restores expected near-harmonic low-mode behavior while keeping symmetric path available explicitly.

## 2026-02-14 Circular Coordinate Paper Alignment
- Timestamp: 2026-02-14T00:00:00Z
- Commit: working tree after spectral gate
- Hypothesis: Align circular-coordinate defaults with paper-style regularization while preserving practical angle quality on the torus benchmark.
- Files touched:
  - `include/igneous/ops/hodge.hpp`
  - `src/main_hodge.cpp`
- Commands:
  - `cmake --build build -j8`
  - `./build/igneous-hodge`
  - `ctest --test-dir build --output-on-failure`
- Baseline:
  - Existing default used `epsilon=1e-3` and identity-style regularization.
- Results:
  - Added additive `CircularCoordinateOptions`.
  - Default epsilon is now `1.0f`.
  - Default path now regularizes using Laplacian-based term with calibrated scale for current discretization.
  - Added deterministic circular diagnostics in `main_hodge` output.
  - Current large torus run now reports:
    - `theta_0 range=[0.000466881, 0.999998], std=0.288768`
    - `theta_1 range=[0.000101466, 0.999627], std=0.281885`
- Numeric checks:
  - `ctest`: pass (`7/7`)
- Decision: `kept`
- Notes:
  - Raw unscaled `epsilon * Delta` over-regularized this discretization and collapsed angular spread; retained Laplacian regularization with explicit scale parameter to preserve paper-aligned structure and practical stability.
