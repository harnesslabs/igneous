#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_ply(path: Path):
    lines = path.read_text().splitlines()
    vertex_count = 0
    header_end = 0
    properties = []

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            if len(parts) == 3:
                properties.append(parts[2])
        elif line.strip() == "end_header":
            header_end = i + 1
            break

    if vertex_count <= 0 or header_end <= 0:
        raise ValueError(f"invalid PLY header in {path}")

    data = np.loadtxt(lines[header_end : header_end + vertex_count], dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    name_to_idx = {name: idx for idx, name in enumerate(properties)}
    points = np.stack(
        [data[:, name_to_idx["x"]], data[:, name_to_idx["y"]], data[:, name_to_idx["z"]]],
        axis=1,
    )

    colors = None
    if {"red", "green", "blue"}.issubset(name_to_idx):
        colors = np.stack(
            [
                data[:, name_to_idx["red"]],
                data[:, name_to_idx["green"]],
                data[:, name_to_idx["blue"]],
            ],
            axis=1,
        ).astype(np.uint8)
    elif {"nx", "ny", "nz"}.issubset(name_to_idx):
        vectors = np.stack(
            [data[:, name_to_idx["nx"]], data[:, name_to_idx["ny"]], data[:, name_to_idx["nz"]]],
            axis=1,
        )
        mag = np.linalg.norm(vectors, axis=1)
        m_min = float(np.min(mag))
        m_max = float(np.max(mag))
        denom = max(m_max - m_min, 1e-12)
        t = (mag - m_min) / denom
        colors = np.stack(
            [
                (255.0 * t).astype(np.uint8),
                (255.0 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8),
                (255.0 * (1.0 - t)).astype(np.uint8),
            ],
            axis=1,
        )

    return points, colors


def make_projection(points, colors, out_path: Path, axes=(0, 1), size=900):
    pad = 20
    image = Image.new("RGB", (size, size), (20, 20, 20))
    draw = ImageDraw.Draw(image)

    a = points[:, axes[0]]
    b = points[:, axes[1]]
    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))

    denom_a = max(a_max - a_min, 1e-12)
    denom_b = max(b_max - b_min, 1e-12)

    px = ((a - a_min) / denom_a * (size - 2 * pad) + pad).astype(np.int32)
    py = ((1.0 - (b - b_min) / denom_b) * (size - 2 * pad) + pad).astype(np.int32)

    if colors is None:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    for x, y, c in zip(px, py, colors):
        draw.point((int(x), int(y)), fill=(int(c[0]), int(c[1]), int(c[2])))

    image.save(out_path)


def summarize_points(points, colors):
    summary = {
        "num_points": int(points.shape[0]),
        "bbox_min": points.min(axis=0).astype(float).tolist(),
        "bbox_max": points.max(axis=0).astype(float).tolist(),
    }
    if colors is not None:
        summary["color_std"] = colors.std(axis=0).astype(float).tolist()
        summary["color_min"] = colors.min(axis=0).astype(int).tolist()
        summary["color_max"] = colors.max(axis=0).astype(int).tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate headless Hodge report artifacts (JSON + PNG quicklooks)."
    )
    parser.add_argument("--input-dir", default="output_hodge")
    parser.add_argument("--output-json", default="output_hodge/report_hodge_metrics.json")
    parser.add_argument("--metrics-file", default="output_hodge/hodge_metrics.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    metrics_file = Path(args.metrics_file)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    runtime_metrics = {}
    if metrics_file.exists():
        runtime_metrics = json.loads(metrics_file.read_text())

    files_to_scan = [
        "torus_angle_0.ply",
        "torus_angle_1.ply",
        "harmonic_form_0.ply",
        "harmonic_form_1.ply",
        "harmonic_form_2.ply",
        "decomp_exact.ply",
        "decomp_harmonic.ply",
        "decomp_coexact.ply",
    ]

    report = {
        "input_dir": str(input_dir),
        "runtime_metrics": runtime_metrics,
        "files": {},
        "checks": {},
    }

    for name in files_to_scan:
        ply_path = input_dir / name
        if not ply_path.exists():
            continue

        points, colors = parse_ply(ply_path)
        stem = ply_path.stem
        png_xy = output_json.parent / f"{stem}_xy.png"
        png_yz = output_json.parent / f"{stem}_yz.png"
        make_projection(points, colors, png_xy, axes=(0, 1))
        make_projection(points, colors, png_yz, axes=(1, 2))

        report["files"][name] = summarize_points(points, colors)
        report["files"][name]["quicklook_xy"] = str(png_xy)
        report["files"][name]["quicklook_yz"] = str(png_yz)

    checks = {}
    evals = runtime_metrics.get("hodge_spectrum_first12", [])
    if len(evals) >= 2:
        checks["harmonic_low_mode_envelope"] = bool(evals[0] < 1e-3 and evals[1] < 1e-3)

    theta0 = runtime_metrics.get("theta_0", {})
    theta1 = runtime_metrics.get("theta_1", {})
    if theta0 and theta1:
        theta0_ok = (
            theta0.get("min", 1.0) < 0.1
            and theta0.get("max", 0.0) > 0.9
            and theta0.get("std", 0.0) > 0.1
        )
        theta1_ok = (
            theta1.get("min", 1.0) < 0.1
            and theta1.get("max", 0.0) > 0.9
            and theta1.get("std", 0.0) > 0.1
        )
        checks["circular_coordinate_dynamic_range"] = bool(theta0_ok and theta1_ok)

    report["checks"] = checks
    output_json.write_text(json.dumps(report, indent=2))

    print(f"[report_hodge] wrote {output_json}")
    if checks:
        print(f"[report_hodge] checks: {checks}")


if __name__ == "__main__":
    main()
