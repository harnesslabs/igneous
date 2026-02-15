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
    parser = argparse.ArgumentParser(description="View outputs from src/main_diffusion.cpp")
    parser.add_argument("--run", action="store_true", help="Run igneous-diffusion before rendering")
    parser.add_argument("--executable", default="build/igneous-diffusion")
    parser.add_argument("--input-obj", default="assets/bunny.obj")
    parser.add_argument("--output-dir", default="output_diffusion")
    parser.add_argument("--plots-dir", default="visualizations/results/main_diffusion")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth heat file")
    parser.add_argument("--open", action="store_true", help="Open generated previews (or raw outputs)")
    args = parser.parse_args()

    root = repo_root()
    exe = resolve_path(args.executable)
    input_obj = resolve_path(args.input_obj)
    output_dir = resolve_path(args.output_dir)
    plots_dir = ensure_dir(args.plots_dir)

    if args.run:
        run_command([str(exe), str(input_obj)], cwd=root)

    files = sorted(output_dir.glob("*.ply"))
    if not files:
        raise SystemExit("No .ply outputs found. Run with --run or verify --output-dir.")

    stride = max(1, args.stride)
    render_files = files[::stride]

    if not has_matplotlib():
        print("[viz] matplotlib missing; listing/opening raw outputs")
        list_sources(render_files)
        if args.open:
            open_paths(render_files)
        return 0

    rows: list[tuple[str, Path, Path]] = []
    for src in render_files:
        out_png = plots_dir / f"{src.stem}.png"
        render_geometry(src, out_png)
        rows.append((src.name, src, out_png))

    index_path = plots_dir / "index.md"
    write_index(index_path, "main_diffusion outputs", rows)
    print(f"[viz] Rendered {len(render_files)} file(s) from {len(files)} total")
    print(f"[viz] Wrote previews to {plots_dir}")
    print(f"[viz] Index: {index_path}")

    if args.open:
        open_paths([plots_dir])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
