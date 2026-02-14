#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def generate_torus(n: int, major_radius: float, minor_radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    return np.column_stack([x, y, z])


def generate_sphere(n: int, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.uniform(0.0, 1.0, size=n)
    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)
    sin_phi = np.sin(phi)
    x = radius * sin_phi * np.cos(theta)
    y = radius * sin_phi * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def write_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        for row in points:
            w.writerow([float(row[0]), float(row[1]), float(row[2])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic manifold point clouds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["torus", "sphere"], default="torus")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--major-radius", type=float, default=2.0)
    parser.add_argument("--minor-radius", type=float, default=1.0)
    parser.add_argument("--sphere-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.kind == "torus":
        points = generate_torus(args.n, args.major_radius, args.minor_radius, args.seed)
    else:
        points = generate_sphere(args.n, args.sphere_radius, args.seed)

    write_csv(Path(args.output), points)


if __name__ == "__main__":
    main()
