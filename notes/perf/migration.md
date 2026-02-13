# API and Data Layout Migration Notes

## Geometry Buffer
- `GeometryBuffer` now uses explicit SoA arrays: `x`, `y`, `z`.
- Removed packed interleaved storage assumptions.

## Triangle Topology
- Hot-path face arrays (`face_v0`, `face_v1`, `face_v2`) are built explicitly.
- Added CSR adjacency for both incident faces and neighboring vertices.

## Kernel APIs
- Curvature and flow kernels now require explicit workspace objects.
- Callers must manage reusable buffers for low-allocation hot loops.
