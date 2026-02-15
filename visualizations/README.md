# Visualization Scripts

Scripts in this directory render outputs from each `src/main_*.cpp` binary.

## Dependency

For PNG previews:

```bash
python3 -m pip install numpy matplotlib
```

Without `matplotlib`, scripts fall back to listing/opening raw files.

## Per-Main Viewers

`src/main_point.cpp`:

```bash
python3 visualizations/view_main_point.py --run --open
```

`src/main_mesh.cpp`:

```bash
python3 visualizations/view_main_mesh.py --run --open
```

`src/main_diffusion.cpp`:

```bash
python3 visualizations/view_main_diffusion.py --run --open
```

`src/main_spectral.cpp`:

```bash
python3 visualizations/view_main_spectral.py --run --open
```

`src/main_hodge.cpp`:

```bash
python3 visualizations/view_main_hodge.py --run --open
```

`src/main_diffusion_topology.cpp`:

```bash
python3 visualizations/view_main_diffusion_topology.py --run --open
```

Each script writes previews and an `index.md` under `visualizations/results/<main_name>/`.
