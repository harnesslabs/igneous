# Wave 3 Report

## Outcome
- Status: active.
- Current focus: large-basis diffusion eigenbasis time first, then remaining diffgeo spectrum/reconstruction overhead.
- Current branch head: `49b3bd3`.

## Kept Improvements
1. `2c56f6d` `perf(diffgeo): add wave3 harness and reuse form operators`
   - Added `bench_diffgeo` and the Wave 3 perf note scaffolding.
   - Reused 1-form operators inside the 2-form path and replaced the inner `2x2` solve with closed-form adjugate arithmetic.
2. `6ca4ba7` `perf(diffgeo): densify weak derivative assembly`
   - Rewrote the generic k-form weak derivative around dense block couplings.
3. `2810304` `perf(spectral): reuse circular coordinate setup`
   - Shared circular-coordinate setup across the paired solves and densified `X_op` assembly.
4. `2a6213d` `perf(diffgeo): parallelize form up-laplacian assembly`
   - Parallelized exact upper-triangle assembly for the generic up-Laplacian sweep.
5. `1425f1f` `perf(spectral): cache normalized kernel operator`
   - Cached the normalized symmetric-kernel CSR lazily on first eigensolve so the solver stops reapplying row normalization inside every matvec.
6. `dee719f` `perf(diffusion): fuse post-knn build passes`
   - Parallelized and fused the post-kNN diffusion-build passes while keeping the accepted sparse symmetrization path intact.
7. `89e82be` `perf(hodge): cache gamma assembly in workspace`
   - Reused mixed gamma and coordinate gamma inside `HodgeWorkspace`, then reset the workspace explicitly in rebuild-heavy loops.
8. `5d0f4db` `perf(dec): parallelize stable face-incidence build`
   - Replaced the large-mesh serial DEC face-incidence count/write passes with a stable block-prefix build that preserves per-vertex face order.
9. `907b8ac` `perf(circular): matrixize workspace preparation`
   - Replaced the mean-centered circular workspace preparation sweeps with exact matrix formulas for `Gamma(x, phi)` and the scalar weak Laplacian.
10. `2a24aef` `perf(spectral): loosen tolerance for small basis solves`
   - Relaxed the Spectra convergence tolerance only for `n_eigenvectors <= 16`, cutting the scalar spectral path without moving the 64-mode hodge/diffgeo guards.
11. `5e07f81` `perf(hodge): matrixize curl energy assembly`
   - Replaced the mean-centered Hodge curl-energy basis-pair sweeps with exact dense block contractions over cached Markov and gamma data.
12. `49b3bd3` `perf(spectral): use cholesky whitening for dense solves`
   - Added a direct Cholesky whitening fast path for dense generalized form/Hodge spectra with exact fallback to the older mass-eigenspace regularization path.

## Rejected Fronts
- Normalized symmetric-kernel precompute during diffusion build.
  - Improved isolated eigenbasis kernels.
  - Failed the build-vs-pipeline trade and regressed `bench_diffusion_build/2000`.
- Circular mass-space standard solve.
  - Changed solver strategy without producing a meaningful circular or pipeline win.
- Symmetric normalized-kernel sort/init variants.
  - `LargestAlge` and `sqrt(row_sums)` seeding improved some scalar spectral cases but regressed higher-rank hodge/diffgeo eigensolve targets.
- Large-mesh atomic DEC face incidence.
  - Improved isolated structure build but hurt large-grid curvature/flow locality enough to lose the overall geometry benchmark.
- Symmetric trivial-mode deflation in the normalized-kernel eigensolver.
  - Did not produce a clear repeatable win across the spectral, hodge, and diffgeo pipeline guards.
- Accelerate-backed circular generalized eigensolve.
  - Barely moved the circular phase and added an Apple-only code path.
- Batched paired circular assembly.
  - Removed duplicated operator setup work on paper, but did not materially reduce the actual circular phase.
- Lower spectral matvec parallel threshold.
  - Forcing worker-pool parallelism inside the CSR matvec regressed the eigensolve-heavy paths.
- Global spectral tolerance relaxation.
  - Helped the 16-mode scalar spectral path but regressed the 64-mode hodge eigensolve.
- Davidson symmetric eigensolver.
  - The JD solver family was dramatically slower than the accepted symmetric Lanczos path on the 64-mode target.
- Matrix diffgeo gamma-data immersion.
  - The exact covariance rewrite for `Gamma(data_coord, immersion_coord)` did not move the end-to-end diffgeo pipeline enough to keep.
- Matrix diffgeo mixed-gamma cache.
  - The mean-centered matrix rewrite for the generic form mixed-gamma cache regressed the end-to-end diffgeo pipeline and was rejected at smoke.

## Final Headline Numbers
Using the Wave 3 artifacts in this directory:
- Diffgeo pipeline:
  - Baseline after Wave 3 harness landed: `bench_diffgeo_pipeline/4000/64/32/32 = 4169.532 ms` in `baseline.md`
  - Current head: `145.730 ms` CPU in `bench_diffgeo_form_spectrum_cholesky_candidate_20260307-025546.txt`
- Diffgeo circular phase:
  - Baseline: `2416.567 ms` in `baseline.md`
  - Current head: `3.821 ms` CPU in `bench_diffgeo_circularprep_candidate_20260306-192320.txt`
- Diffgeo eigenbasis phase:
  - Baseline after the harness landed: `893.912 ms` in `baseline.md`
  - Current head: `107.411 ms` CPU in `bench_diffgeo_form_spectrum_cholesky_candidate_20260307-025546.txt`
- Hodge pipeline:
  - Current head: `149.293 ms` CPU in `bench_pipelines_form_spectrum_cholesky_candidate_20260307-025546.txt`
- Hodge circular phase:
  - Current head: `4.034 ms` CPU in `bench_pipelines_hodge_curl_matrix_candidate_20260307-024447.txt`
- Hodge eigenbasis phase:
  - Current head: `115.785 ms` CPU in `bench_pipelines_form_spectrum_cholesky_candidate_20260307-025546.txt`
- Hodge curl-energy phase:
  - Current head: `10.199 ms` CPU in `bench_pipelines_hodge_curl_matrix_candidate_20260307-024447.txt`
- Hodge dense solve phase:
  - Current head: `2.151 ms` CPU in `bench_pipelines_form_spectrum_cholesky_candidate_20260307-025546.txt`
- Spectral pipeline:
  - Current head: `16.783 ms` CPU in `bench_pipelines_small_spectral_tol_candidate_20260307-023622.txt`
- Small spectral eigensolve:
  - `bench_eigenbasis/2000/16`: `11.368 ms` -> `8.199 ms` CPU
- App guard:
  - `igneous-diffusion-geometry --n-points 1000 --output-dir <tempdir>`
  - Early Wave 3 head: `real 1.40`
  - Current head: `real 0.07`
  - `igneous-hodge --no-ply`: current head `real 0.05`
- Geometry large-grid guard:
  - `bench_geometry` `Grid 1000x1000`
  - structure: `17.288 ms -> 14.532 ms`
  - curvature: `5.450 ms -> 5.048 ms`
  - flow: `0.904 ms -> 0.879 ms`

## Current Hotspot Order
### Diffgeo
- `eigenbasis`
- remaining spectrum/reconstruction overhead
- remaining misc pipeline overhead

### Hodge
- `eigenbasis`
- remaining spectrum overhead

## Assessment
- The high-confidence, behavior-preserving wins in diffusion build, generic form assembly, and Hodge gamma reuse have been taken.
- Build-heavy workloads are no longer the main problem.
- The matrix circular-prep rewrite removed the dominant remaining circular hotspot without changing solver behavior, which reset the hotspot ranking.
- The small-basis spectral path still had oversolve headroom, and relaxing its convergence target reclaimed that without moving the higher-priority 64-mode guards.
- The exact matrix curl-energy rewrite removed most of the remaining non-eigenbasis Hodge cost without moving the eigensolve or circular guards.
- The dense generalized-solve fast path now removes a large fraction of the remaining Hodge solve overhead while keeping the diffgeo pipeline within guard.
- The main remaining shared cost is still the large-basis diffusion eigenbasis solve; after that, the pipelines are mostly diffgeo spectrum/reconstruction overhead rather than Hodge or circular assembly.
- The new DEC face-incidence rewrite and the circular-prep rewrite both came from replacing large scalar sweeps with exact matrix formulations that preserve downstream behavior.

## Recommended Next Wave
- Only continue if one of these is acceptable:
  - a new eigensolver trade that improves diffusion eigenbasis without shifting too much cost back into diffusion build,
  - a mathematically exact reduction of the remaining scalar spectrum/reconstruction overhead,
  - or another order-preserving data-oriented rewrite outside diffusion geometry like the accepted DEC face-incidence pass.

## Post-PR Continuation

### Kept Since The PR Was Opened
- `cf653c4` `ci(perf): add diffgeo coverage to smoke and deep reports`
- pending `perf(dec): specialize curvature traversal for DEC storage`

### Current Additional Gains
- DEC curvature:
  - `bench_curvature_kernel/400`: `888653 ns -> 816836 ns` CPU (`-8.08%`)
  - `bench_geometry` `Grid 1000x1000` curvature: `5.716 ms -> 5.358 ms` (`-6.26%`)
  - `bench_geometry` `Grid 1000x1000` structure: `14.516 ms -> 13.661 ms` (`-5.89%`)

### New Rejections
- Weighted coordinate seed for the large-basis symmetric eigensolve.
- CPU Markov multi-step copy-avoidance rewrite.
- Early serial short-circuit in `parallel_for_index`.

### Current Read Of The Remaining Work
- The large sparse diffusion eigensolve is still the biggest shared runtime cost.
- The geometry/DEC path still has room for order-preserving data-oriented rewrites, but the wins are now single-digit rather than order-of-magnitude.
- GitHub perf smoke/deep will now surface `bench_diffgeo` even when the baseline branch predates that benchmark target.
