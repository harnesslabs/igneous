# Wave 3 Report

## Outcome
- Status: paused after the latest kept hodge pass.
- Stop reason: the remaining dominant cost is now circular plus eigensolve, and the next credible wins appear to require more invasive solver/formulation changes than the high-confidence internal refactors used in this wave.
- Current branch head: `89e82be`.

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

## Final Headline Numbers
Using the Wave 3 artifacts in this directory:
- Diffgeo pipeline:
  - Baseline after Wave 3 harness landed: `bench_diffgeo_pipeline/4000/64/32/32 = 4169.532 ms` in `baseline.md`
  - Current head: `404.855 ms` CPU in the current three-repetition rerank
- Diffgeo circular phase:
  - Baseline: `2416.567 ms` in `baseline.md`
  - Current head: `268.336 ms` in the current three-repetition rerank
- Diffgeo eigenbasis phase:
  - Baseline after the harness landed: `893.912 ms` in `baseline.md`
  - Current head: `106.225 ms` in the current three-repetition rerank
- Hodge pipeline:
  - Current head: `423.551 ms` in the current three-repetition rerank
- Hodge circular phase:
  - Current head: `266.806 ms` in the current three-repetition rerank
- Hodge eigenbasis phase:
  - Current head: `117.306 ms` in the current three-repetition rerank
- Spectral pipeline:
  - Current head: `22.367 ms` in the current three-repetition rerank
- App guard:
  - `igneous-diffusion-geometry --n-points 1000 --output-dir <tempdir>`
  - Early Wave 3 head: `real 1.40`
  - Current head: `real 0.11`
  - `igneous-hodge --no-ply`: current head `real 0.10`

## Current Hotspot Order
### Diffgeo
- `circular`
- `eigenbasis`
- remaining spectrum/reconstruction overhead
- remaining misc pipeline overhead

### Hodge
- `circular`
- `eigenbasis`
- remaining spectrum/curl overhead

## Assessment
- The high-confidence, behavior-preserving wins in diffusion build, generic form assembly, and Hodge gamma reuse have been taken.
- Build-heavy workloads are no longer the main problem.
- Both the hodge and diffgeo pipelines are now dominated by circular-coordinate solves and scalar eigensolve time.
- The most recent rejected experiments were on symmetric eigensolver control and DEC structure locality, which is a useful signal that the easy internal refactors are mostly exhausted.

## Recommended Next Wave
- Only continue if one of these is acceptable:
  - a more invasive circular-coordinate formulation or solver change,
  - a different spectral build/eigensolver trade that may move cost between build and solve,
  - API-level caching/lifecycle changes above the current internal-workspace approach,
  - or a new benchmark target for the remaining diffgeo spectrum/reconstruction overhead so that those phases can be attacked directly.
- Otherwise, keep the current head as the Wave 3 endpoint.
