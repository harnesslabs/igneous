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
