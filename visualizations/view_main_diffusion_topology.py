#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    ensure_dir,
    has_matplotlib,
    list_sources,
    open_paths,
    render_geometry,
    repo_root,
    resolve_path,
    run_command,
    write_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="View outputs from src/main_diffusion_topology.cpp"
    )
    parser.add_argument("--run", action="store_true", help="Run igneous-diffusion-topology before rendering")
    parser.add_argument("--executable", default="build/igneous-diffusion-topology")
    parser.add_argument("--output-dir", default="output_diffusion_topology")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--generate-sphere", action="store_true")
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plots-dir", default="visualizations/results/main_diffusion_topology")
    parser.add_argument("--open", action="store_true", help="Open generated previews (or raw outputs)")
    args = parser.parse_args()

    root = repo_root()
    exe = resolve_path(args.executable)
    output_dir = resolve_path(args.output_dir)
    plots_dir = ensure_dir(args.plots_dir)

    if args.run:
        cmd = [str(exe), "--output-dir", str(output_dir)]
        if args.input_csv:
            cmd.extend(["--input-csv", str(resolve_path(args.input_csv))])
        if args.generate_sphere:
            cmd.append("--generate-sphere")
        if args.n_points is not None:
            cmd.extend(["--n-points", str(args.n_points)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        run_command(cmd, cwd=root)

    files = sorted(output_dir.glob("*.ply"))
    if not files:
        raise SystemExit(
            "No diffusion-topology PLY outputs found. Run with --run or verify --output-dir."
        )

    if not has_matplotlib():
        print("[viz] matplotlib missing; listing/opening raw outputs")
        list_sources(files)
        if args.open:
            open_paths(files)
        return 0

    rows: list[tuple[str, Path, Path]] = []
    for src in files:
        out_png = plots_dir / f"{src.stem}.png"
        render_geometry(src, out_png)
        rows.append((src.name, src, out_png))

    index_path = plots_dir / "index.md"
    write_index(index_path, "main_diffusion_topology outputs", rows)
    print(f"[viz] Wrote previews to {plots_dir}")
    print(f"[viz] Index: {index_path}")

    if args.open:
        open_paths([plots_dir])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
