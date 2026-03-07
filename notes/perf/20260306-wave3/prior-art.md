# Prior Art

This campaign should build on previous accepted optimizations and avoid repeating rejected paths.

## Preserve
- CSR-backed diffusion operators and direct Markov traversal.
- Current threaded worker-pool backend and conservative GPU gating.
- Symmetric-kernel eigensolver path and current compact-plus-fallback Arnoldi policy.
- Corrected generic k-form semantics from the diffusion-geometry parity work.
- Shared circular-coordinate setup reuse within a single pipeline call; keep the existing mode ordering and selection semantics unchanged.
- Parallel upper-triangle assembly for generic form up-Laplacians; it preserved outputs and removed the serial bottleneck in the basis-pair sweep.
- Parallelized/fused post-kNN diffusion-build passes while keeping the existing Eigen sparse symmetrization path; this is the accepted build optimization over the rejected direct CSR rewrite.
- Hodge workspace gamma-cache reuse within a fixed mesh/eigenbasis state; if the workspace is reused across rebuilds, reset it explicitly first.
- Stable block-prefix DEC face-incidence construction for large meshes; it preserves per-vertex face order while removing the large-mesh serial bottleneck.
- Matrix-form circular workspace preparation on the mean-centered diffusion path; keep the exact covariance formulas for `Gamma(x, phi)` and the scalar weak Laplacian instead of reverting to pairwise scalar `carre_du_champ` sweeps.
- Relaxed Spectra tolerance only for small-basis solves (`n_eigenvectors <= 16`); the 16-mode scalar spectral path benefits, but the 64-mode hodge/diffgeo guards should stay on the stricter convergence target.
- Matrix-form Hodge curl-energy assembly on the mean-centered path; keep the exact `U^T diag(P^T c) U - (P U)^T diag(c) (P U)` formulation plus the weighted mixed-gamma cross term instead of the old basis-pair `carre_du_champ` sweeps.
- Cholesky whitening for dense generalized form/Hodge spectra with fallback to the old mass-eigenspace path; this is the accepted fast path for the current SPD mass matrices.

## Do Not Repeat
- Tightening Arnoldi subspace below the current accepted compact fallback.
- Circular-coordinate solver/order changes that alter mode selection.
- KD-tree leaf-size experiments beyond the accepted setting.
- Scratch-reuse-only attempts in diffusion build.
- Hybrid brute-force kNN variants.
- Naive circular-coordinate cache reuse.
- The rejected 1-form block-GEMM rewrite as-is.
- Low-value curl-energy micro-tweaks that only move isolated kernels without pipeline wins.
- Precomputing normalized symmetric-kernel CSR values during diffusion build; it improved isolated eigenbasis solves but failed the build-vs-pipeline trade.
- Replacing the circular generalized solve with a cached mass-space standard solve; it only moved the phase at noise level and regressed the diffgeo pipeline.
- Deflating the trivial normalized-kernel diffusion mode inside the symmetric Spectra path; it did not produce a clear, repeatable pipeline win.
- Replacing the circular generalized eigensolver with the macOS Accelerate/CLAPACK path; it only moved the circular phase at noise level and added platform-specific complexity.
- Batching the paired circular-operator assembly; it did not materially reduce the remaining circular phase.
- The direct CSR symmetrization rewrite tested in Wave 3 improved some structure phases but regressed downstream spectral/pipeline behavior; do not reuse that exact formulation.
- Treat the `4000/64` diffgeo torus eigensolve as a guard when changing diffusion build internals; the accepted post-kNN fusion pass already spends some of that headroom.
- Lowering the spectral CSR matvec parallel threshold to force parallel row sweeps at `n ~= 2k-4k`; the worker-pool overhead regressed the eigensolve-heavy paths.
- Relaxing the Spectra tolerance globally on large-basis solves; it helps the 16-mode scalar spectral path but regresses the 64-mode hodge guard.
- Replacing the symmetric normalized-kernel solver with Spectra's Davidson path; it remained much slower than the accepted Lanczos solver even with a direct CSR adapter.
- Matrixizing `Gamma(data_coord, immersion_coord)` for diffgeo reconstruction; the diffgeo pipeline stayed effectively flat, so that precompute is not worth revisiting on the current head.

## New Focus
- Benchmark and optimize the generic k-form pipeline, which was added after most of the earlier perf work.
- Reuse already-computed 1-form operators inside the 2-form path.
- Remove exact recomputation before trying lower-level loop tuning.
- Preserve the lazy normalized symmetric-kernel cache inside eigensolve; it is the accepted spectral improvement over the rejected build-time precompute.
- Revisit diffusion build, circular solve internals, or non-diffusion geometry kernels only if they can beat the current head without shifting cost into downstream spectral behavior.
- Treat the 16-mode scalar spectral path separately from the 64-mode hodge/diffgeo guard path when changing solver tolerances or policy.
- Treat dense generalized spectra separately from the large sparse diffusion eigensolve; the dense form/Hodge solve now has an accepted Cholesky fast path and should not be conflated with the sparse basis extraction problem.
