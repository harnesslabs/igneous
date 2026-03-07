# Prior Art

This campaign should build on previous accepted optimizations and avoid repeating rejected paths.

## Preserve
- CSR-backed diffusion operators and direct Markov traversal.
- Current threaded worker-pool backend and conservative GPU gating.
- Symmetric-kernel eigensolver path and current compact-plus-fallback Arnoldi policy.
- Corrected generic k-form semantics from the diffusion-geometry parity work.
- Shared circular-coordinate setup reuse within a single pipeline call; keep the existing mode ordering and selection semantics unchanged.

## Do Not Repeat
- Tightening Arnoldi subspace below the current accepted compact fallback.
- Circular-coordinate solver/order changes that alter mode selection.
- KD-tree leaf-size experiments beyond the accepted setting.
- Scratch-reuse-only attempts in diffusion build.
- Hybrid brute-force kNN variants.
- Naive circular-coordinate cache reuse.
- The rejected 1-form block-GEMM rewrite as-is.
- Low-value curl-energy micro-tweaks that only move isolated kernels without pipeline wins.

## New Focus
- Benchmark and optimize the generic k-form pipeline, which was added after most of the earlier perf work.
- Reuse already-computed 1-form operators inside the 2-form path.
- Remove exact recomputation before trying lower-level loop tuning.
- Revisit `k2_up`, scalar diffusion eigenbasis, or diffusion build only if they remain dominant after the circular pass.
