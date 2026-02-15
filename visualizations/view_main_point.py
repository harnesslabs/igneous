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
    parser = argparse.ArgumentParser(description="View outputs from src/main_point.cpp")
    parser.add_argument("--run", action="store_true", help="Run igneous-point before rendering")
    parser.add_argument("--executable", default="build/igneous-point")
    parser.add_argument("--input-obj", default="assets/bunny.obj")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--plots-dir", default="visualizations/results/main_point")
    parser.add_argument("--open", action="store_true", help="Open generated previews (or raw outputs)")
    args = parser.parse_args()

    root = repo_root()
    exe = resolve_path(args.executable)
    input_obj = resolve_path(args.input_obj)
    output_root = resolve_path(args.output_root)
    plots_dir = ensure_dir(args.plots_dir)

    if args.run:
        run_command([str(exe), str(input_obj)], cwd=root)

    artifacts = [
        output_root / "output_solid_cloud.ply",
        output_root / "output_surface.obj",
    ]
    artifacts = [p for p in artifacts if p.exists()]
    if not artifacts:
        raise SystemExit("No main_point outputs found. Run with --run or verify paths.")

    if not has_matplotlib():
        print("[viz] matplotlib missing; listing/opening raw outputs")
        list_sources(artifacts)
        if args.open:
            open_paths(artifacts)
        return 0

    rows: list[tuple[str, Path, Path]] = []
    for art in artifacts:
        out_png = plots_dir / f"{art.stem}.png"
        render_geometry(art, out_png)
        rows.append((art.name, art, out_png))

    index_path = plots_dir / "index.md"
    write_index(index_path, "main_point outputs", rows)
    print(f"[viz] Wrote previews to {plots_dir}")
    print(f"[viz] Index: {index_path}")

    if args.open:
        open_paths([plots_dir])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
