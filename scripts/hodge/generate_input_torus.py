#!/usr/bin/env python3
import argparse
import csv
import pathlib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic torus point cloud CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--major-radius", type=float, default=2.0)
    parser.add_argument("--minor-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    u = rng.uniform(0.0, 2.0 * np.pi, size=args.n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=args.n)

    x = (args.major_radius + args.minor_radius * np.cos(v)) * np.cos(u)
    y = (args.major_radius + args.minor_radius * np.cos(v)) * np.sin(u)
    z = args.minor_radius * np.sin(v)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["x", "y", "z"])
      for row in np.stack([x, y, z], axis=1):
          writer.writerow([float(row[0]), float(row[1]), float(row[2])])


if __name__ == "__main__":
    main()
