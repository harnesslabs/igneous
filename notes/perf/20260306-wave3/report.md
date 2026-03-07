# Wave 3 Report

## Outcome
- Status: active.
- Current focus: diffusion eigenbasis time first, then remaining spectrum/reconstruction overhead.
- Current branch head: `907b8ac`.

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

## Final Headline Numbers
Using the Wave 3 artifacts in this directory:
- Diffgeo pipeline:
  - Baseline after Wave 3 harness landed: `bench_diffgeo_pipeline/4000/64/32/32 = 4169.532 ms` in `baseline.md`
  - Current head: `147.268 ms` CPU in `bench_diffgeo_circularprep_candidate_20260306-192320.txt`
- Diffgeo circular phase:
  - Baseline: `2416.567 ms` in `baseline.md`
  - Current head: `3.821 ms` CPU in `bench_diffgeo_circularprep_candidate_20260306-192320.txt`
- Diffgeo eigenbasis phase:
  - Baseline after the harness landed: `893.912 ms` in `baseline.md`
  - Current head: `107.143 ms` CPU in `bench_diffgeo_circularprep_candidate_20260306-192320.txt`
- Hodge pipeline:
  - Current head: `164.629 ms` CPU in `bench_pipelines_circularprep_candidate_20260306-192320.txt`
- Hodge circular phase:
  - Current head: `4.074 ms` CPU in `bench_pipelines_circularprep_candidate_20260306-192320.txt`
- Hodge eigenbasis phase:
  - Current head: `116.807 ms` CPU in `bench_pipelines_circularprep_candidate_20260306-192320.txt`
- Spectral pipeline:
  - Current head: `22.367 ms` in the current three-repetition rerank
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
- remaining spectrum/curl overhead

## Assessment
- The high-confidence, behavior-preserving wins in diffusion build, generic form assembly, and Hodge gamma reuse have been taken.
- Build-heavy workloads are no longer the main problem.
- The matrix circular-prep rewrite removed the dominant remaining circular hotspot without changing solver behavior, which reset the hotspot ranking.
- The main remaining shared cost is diffusion eigenbasis time; after that, the pipelines are mostly spectrum/reconstruction overhead rather than circular-coordinate assembly.
- The new DEC face-incidence rewrite and the circular-prep rewrite both came from replacing large scalar sweeps with exact matrix formulations that preserve downstream behavior.

## Recommended Next Wave
- Only continue if one of these is acceptable:
  - a new eigensolver trade that improves diffusion eigenbasis without shifting too much cost back into diffusion build,
  - a mathematically exact reduction of the remaining scalar spectrum/reconstruction overhead,
  - or another order-preserving data-oriented rewrite outside diffusion geometry like the accepted DEC face-incidence pass.
