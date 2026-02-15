#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_matplotlib() -> bool:
    return plt is not None


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("[viz] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_ascii_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = f.readline().strip()
        if "ascii" not in fmt:
            raise ValueError(f"Only ASCII PLY supported: {path}")

        current_element = ""
        vertex_count = 0
        vertex_props: list[str] = []

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Missing end_header in {path}")
            line = line.strip()
            if line.startswith("comment"):
                continue
            if line.startswith("element "):
                parts = line.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                    vertex_props = []
                continue
            if line.startswith("property ") and current_element == "vertex":
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_props.append(parts[2])
                continue
            if line == "end_header":
                break

        if vertex_count <= 0:
            return {}

        rows = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            vals = line.strip().split()
            if len(vals) < len(vertex_props):
                continue
            rows.append([float(v) for v in vals[: len(vertex_props)]])

    if not rows:
        return {}

    arr = np.asarray(rows, dtype=np.float64)
    return {name: arr[:, i] for i, name in enumerate(vertex_props)}


def _read_obj_vertices(path: Path) -> dict[str, np.ndarray]:
    xyz: list[list[float]] = []
    rgb: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            xyz.append([x, y, z])
            if len(parts) >= 7:
                rgb.append([float(parts[4]), float(parts[5]), float(parts[6])])

    if not xyz:
        return {}

    xyz_arr = np.asarray(xyz, dtype=np.float64)
    out = {"x": xyz_arr[:, 0], "y": xyz_arr[:, 1], "z": xyz_arr[:, 2]}

    if len(rgb) == len(xyz):
        rgb_arr = np.asarray(rgb, dtype=np.float64)
        if np.max(rgb_arr) > 1.0:
            rgb_arr = rgb_arr / 255.0
        out["red"] = np.clip(rgb_arr[:, 0], 0.0, 1.0)
        out["green"] = np.clip(rgb_arr[:, 1], 0.0, 1.0)
        out["blue"] = np.clip(rgb_arr[:, 2], 0.0, 1.0)

    return out


def _set_equal_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    spans = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    max_span = float(np.max(spans))
    if max_span <= 0.0:
        max_span = 1.0
    cx = 0.5 * (x.max() + x.min())
    cy = 0.5 * (y.max() + y.min())
    cz = 0.5 * (z.max() + z.min())
    h = 0.5 * max_span
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy - h, cy + h)
    ax.set_zlim(cz - h, cz + h)


def render_geometry(path: Path, out_png: Path, max_arrows: int = 350) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    ext = path.suffix.lower()
    if ext == ".ply":
        data = _read_ascii_ply(path)
    elif ext == ".obj":
        data = _read_obj_vertices(path)
    else:
        raise ValueError(f"Unsupported geometry file: {path}")

    if not data or not {"x", "y", "z"}.issubset(data):
        raise ValueError(f"No renderable vertices in {path}")

    x = data["x"]
    y = data["y"]
    z = data["z"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if {"nx", "ny", "nz"}.issubset(data):
        nx = data["nx"]
        ny = data["ny"]
        nz = data["nz"]
        n = x.shape[0]
        step = max(1, n // max(1, max_arrows))
        idx = np.arange(0, n, step, dtype=np.int64)

        xx = x[idx]
        yy = y[idx]
        zz = z[idx]
        ux = nx[idx]
        uy = ny[idx]
        uz = nz[idx]

        mag = np.sqrt(ux * ux + uy * uy + uz * uz)
        nonzero = mag > 1e-12
        ux = np.where(nonzero, ux / np.maximum(mag, 1e-12), 0.0)
        uy = np.where(nonzero, uy / np.maximum(mag, 1e-12), 0.0)
        uz = np.where(nonzero, uz / np.maximum(mag, 1e-12), 0.0)

        spans = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
        diag = float(np.linalg.norm(spans))
        if diag <= 0.0:
            diag = 1.0
        arrow_len = 0.08 * diag

        ax.scatter(xx, yy, zz, s=2, c="black", alpha=0.25, linewidths=0)
        ax.quiver(xx, yy, zz, ux, uy, uz, length=arrow_len, normalize=False, linewidth=0.6)
    else:
        if {"red", "green", "blue"}.issubset(data):
            c = np.column_stack([data["red"], data["green"], data["blue"]])
            if c.max() > 1.0:
                c = c / 255.0
        else:
            c = z
        ax.scatter(x, y, z, c=c, s=4, linewidths=0)

    _set_equal_axes(ax, x, y, z)
    ax.set_title(path.name)
    ax.set_axis_off()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows: list[list[float]] = []
        for row in reader:
            vals = []
            for h in headers:
                v = row.get(h, "")
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            rows.append(vals)

    if not headers:
        return [], np.zeros((0, 0), dtype=np.float64)

    if not rows:
        return headers, np.zeros((0, len(headers)), dtype=np.float64)

    return headers, np.asarray(rows, dtype=np.float64)


def render_csv_preview(path: Path, out_png: Path, max_arrows: int = 350) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    headers, values = _read_numeric_csv(path)
    if not headers:
        raise ValueError(f"No CSV header in {path}")

    idx = {h: i for i, h in enumerate(headers)}

    fig = plt.figure(figsize=(8, 7))

    if "mode" in idx and "lambda" in idx and values.shape[0] > 0:
        ax = fig.add_subplot(111)
        ax.plot(values[:, idx["mode"]], values[:, idx["lambda"]], linewidth=1.5)
        ax.set_xlabel("mode")
        ax.set_ylabel("lambda")
        ax.grid(alpha=0.3)
    elif {"x", "y", "z"}.issubset(idx):
        x = values[:, idx["x"]]
        y = values[:, idx["y"]]
        z = values[:, idx["z"]]
        ax = fig.add_subplot(111, projection="3d")

        vector_triplet = None
        for h in headers:
            if not h.endswith("_x"):
                continue
            stem = h[:-2]
            if f"{stem}_y" in idx and f"{stem}_z" in idx:
                vector_triplet = stem
                break

        if vector_triplet is not None:
            ux = values[:, idx[f"{vector_triplet}_x"]]
            uy = values[:, idx[f"{vector_triplet}_y"]]
            uz = values[:, idx[f"{vector_triplet}_z"]]
            n = x.shape[0]
            step = max(1, n // max(1, max_arrows))
            sel = np.arange(0, n, step, dtype=np.int64)

            xx = x[sel]
            yy = y[sel]
            zz = z[sel]
            ux = ux[sel]
            uy = uy[sel]
            uz = uz[sel]

            mag = np.sqrt(ux * ux + uy * uy + uz * uz)
            nonzero = mag > 1e-12
            ux = np.where(nonzero, ux / np.maximum(mag, 1e-12), 0.0)
            uy = np.where(nonzero, uy / np.maximum(mag, 1e-12), 0.0)
            uz = np.where(nonzero, uz / np.maximum(mag, 1e-12), 0.0)

            spans = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
            diag = float(np.linalg.norm(spans))
            if diag <= 0.0:
                diag = 1.0
            arrow_len = 0.08 * diag

            ax.scatter(xx, yy, zz, s=2, c="black", alpha=0.25, linewidths=0)
            ax.quiver(xx, yy, zz, ux, uy, uz, length=arrow_len, normalize=False, linewidth=0.6)
        else:
            scalar_cols = [h for h in headers if h not in {"x", "y", "z"}]
            if scalar_cols:
                c = values[:, idx[scalar_cols[0]]]
                ax.scatter(x, y, z, c=c, s=4, linewidths=0, cmap="viridis")
            else:
                ax.scatter(x, y, z, s=4, linewidths=0)

        _set_equal_axes(ax, x, y, z)
        ax.set_axis_off()
    elif values.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.plot(values[:, 0], values[:, 1], linewidth=1.5)
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.grid(alpha=0.3)
    else:
        ax = fig.add_subplot(111)
        mat = values
        if mat.size == 0:
            mat = np.zeros((1, 1), dtype=np.float64)
        if mat.shape[0] > 400:
            mat = mat[:400, :]
        if mat.shape[1] > 400:
            mat = mat[:, :400]
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    fig.suptitle(path.name)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_index(markdown_path: Path, title: str, rows: list[tuple[str, Path, Path]]) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| Artifact | Source | Preview |")
    lines.append("|---|---|---|")
    for label, source, preview in rows:
        rel_src = source
        rel_prev = preview
        lines.append(f"| {label} | `{rel_src}` | ![]({rel_prev}) |")
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def list_sources(paths: Iterable[Path]) -> None:
    for p in paths:
        print(f"[viz] {p}")


def open_paths(paths: Iterable[Path]) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener) is None:
        return
    for p in paths:
        subprocess.run([opener, str(p)], check=False)
