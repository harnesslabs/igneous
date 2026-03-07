# Wave 3 Report

## Outcome
- Status: paused after hitting the campaign stop rule.
- Stop reason: two consecutive major fronts failed to produce a keep-worthy change after the last kept diffgeo win.
- Current branch head: `d1db951`.

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

## Rejected Fronts
- Normalized symmetric-kernel precompute during diffusion build.
  - Improved isolated eigenbasis kernels.
  - Failed the build-vs-pipeline trade and regressed `bench_diffusion_build/2000`.
- Circular mass-space standard solve.
  - Changed solver strategy without producing a meaningful circular or pipeline win.

## Final Headline Numbers
Using the Wave 3 artifacts in this directory:
- Diffgeo pipeline:
  - Baseline after Wave 3 harness landed: `bench_diffgeo_pipeline/4000/64/32/32 = 4169.532 ms` in `baseline.md`
  - Current best kept result: `444.296 ms` CPU in `results/bench_diffgeo_up_parallel_20260306-173505.txt`
- Diffgeo circular phase:
  - Baseline: `2416.567 ms` in `baseline.md`
  - After circular reuse: `274.929 ms` in `results/bench_diffgeo_circular_20260306-172407.txt`
  - Current head: `272.185 ms` in `results/bench_diffgeo_up_parallel_20260306-173505.txt`
- Diffgeo up terms:
  - `bench_diffgeo_phase_k1_up/4000/64/32/32`: `73.024 ms` -> `6.498 ms`
  - `bench_diffgeo_phase_k2_up/4000/64/32/32`: `102.051 ms` -> `8.692 ms`
- App guard:
  - `igneous-diffusion-geometry --n-points 1000 --output-dir <tempdir>`
  - Early Wave 3 head: `real 1.40`
  - After the circular reuse pass: `real 0.24`
  - Current head after the up-Laplacian pass: `real 0.13`

## Current Hotspot Order
### Diffgeo
- `circular`
- `eigenbasis`
- `structure_build`
- remaining misc pipeline overhead

### Hodge
- `circular`
- `eigenbasis`
- `structure_build`

## Assessment
- The high-confidence, behavior-preserving wins in the generic diffusion-form path have been taken.
- The remaining big items are circular and scalar eigenbasis.
- The last two serious attempts on those fronts failed the campaign gate without changing semantics.
- Further gains are still possible, but they are less likely to come from internal refactors alone.

## Recommended Next Wave
- Only continue if one of these is acceptable:
  - a more invasive circular-coordinate formulation or solver change,
  - a different spectral build/eigensolver trade that may move cost between build and solve,
  - API-level caching/lifecycle changes above the current internal-workspace approach.
- Otherwise, keep the current head as the Wave 3 endpoint.
