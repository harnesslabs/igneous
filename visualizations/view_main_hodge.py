#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    ensure_dir,
    has_matplotlib,
    list_sources,
    open_paths,
    render_csv_preview,
    render_geometry,
    repo_root,
    resolve_path,
    run_command,
    write_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="View outputs from src/main_hodge.cpp")
    parser.add_argument("--run", action="store_true", help="Run igneous-hodge before rendering")
    parser.add_argument("--executable", default="build/igneous-hodge")
    parser.add_argument("--output-dir", default="output_hodge")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--plots-dir", default="visualizations/results/main_hodge")
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
        if args.n_points is not None:
            cmd.extend(["--n-points", str(args.n_points)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.no_ply:
            cmd.append("--no-ply")
        run_command(cmd, cwd=root)

    csv_files = sorted(output_dir.glob("*.csv"))
    ply_files = sorted(output_dir.glob("*.ply"))
    artifacts = [*csv_files, *ply_files]
    if not artifacts:
        raise SystemExit("No hodge outputs found. Run with --run or verify --output-dir.")

    if not has_matplotlib():
        print("[viz] matplotlib missing; listing/opening raw outputs")
        list_sources(artifacts)
        if args.open:
            open_paths(artifacts)
        return 0

    rows: list[tuple[str, Path, Path]] = []

    for src in csv_files:
        out_png = plots_dir / f"{src.stem}_csv.png"
        render_csv_preview(src, out_png)
        rows.append((src.name, src, out_png))

    for src in ply_files:
        out_png = plots_dir / f"{src.stem}_ply.png"
        render_geometry(src, out_png)
        rows.append((src.name, src, out_png))

    index_path = plots_dir / "index.md"
    write_index(index_path, "main_hodge outputs", rows)
    print(f"[viz] Wrote previews to {plots_dir}")
    print(f"[viz] Index: {index_path}")

    if args.open:
        open_paths([plots_dir])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
