#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return np.genfromtxt(path, delimiter=",", names=True)


def wrapped_angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def ambient_form_indices(dtype_names: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    i = 0
    while f"form{i}_x" in dtype_names:
        if f"form{i}_y" in dtype_names and f"form{i}_z" in dtype_names:
            out.append(i)
        i += 1
    return out


def theta_indices(dtype_names: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    i = 0
    while f"theta_{i}" in dtype_names:
        out.append(i)
        i += 1
    return out


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    step = max(1, n // max_points)
    return np.arange(0, n, step, dtype=np.int64)[:max_points]


def set_equal_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    mins = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    maxs = np.array([x.max(), y.max(), z.max()], dtype=np.float64)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_vector_pair(
    out_path: Path,
    points_ref: np.ndarray,
    vec_ref: np.ndarray,
    points_cpp: np.ndarray,
    vec_cpp: np.ndarray,
    title: str,
):
    fig = plt.figure(figsize=(14, 6))
    ax_ref = fig.add_subplot(1, 2, 1, projection="3d")
    ax_cpp = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, pts, vecs, label in [
        (ax_ref, points_ref, vec_ref, "Reference"),
        (ax_cpp, points_cpp, vec_cpp, "C++"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c="lightgray", alpha=0.35)
        ax.quiver(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            vecs[:, 0],
            vecs[:, 1],
            vecs[:, 2],
            length=0.18,
            normalize=True,
            linewidth=0.5,
            color="#005f73",
        )
        set_equal_axes(ax, pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_angle_pair(
    out_path: Path,
    points_ref: np.ndarray,
    theta_ref: np.ndarray,
    points_cpp: np.ndarray,
    theta_cpp: np.ndarray,
    title: str,
):
    fig = plt.figure(figsize=(14, 6))
    ax_ref = fig.add_subplot(1, 2, 1, projection="3d")
    ax_cpp = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, pts, theta, label in [
        (ax_ref, points_ref, theta_ref, "Reference"),
        (ax_cpp, points_cpp, theta_cpp, "C++"),
    ]:
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=theta,
            s=9,
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * np.pi,
        )
        set_equal_axes(ax, pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.03)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_circular_forms_grid(
    out_path: Path,
    points_ref: np.ndarray,
    points_cpp: np.ndarray,
    theta_ref0: np.ndarray,
    theta_cpp0: np.ndarray,
    theta_ref1: np.ndarray,
    theta_cpp1: np.ndarray,
):
    fig = plt.figure(figsize=(14, 12))
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    panels = [
        (axes[0], points_ref, theta_ref0, "Reference: Circular from Harmonic 1-Form 0"),
        (axes[1], points_cpp, theta_cpp0, "C++: Circular from Harmonic 1-Form 0"),
        (axes[2], points_ref, theta_ref1, "Reference: Circular from Harmonic 1-Form 1"),
        (axes[3], points_cpp, theta_cpp1, "C++: Circular from Harmonic 1-Form 1"),
    ]

    for ax, pts, theta, label in panels:
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=theta,
            s=8,
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * np.pi,
        )
        set_equal_axes(ax, pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.colorbar(sc, ax=ax, shrink=0.58, pad=0.02)

    fig.suptitle("Circular Coordinates by Source Harmonic 1-Form", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_angle_error_hist(
    out_path: Path, errors: np.ndarray, title: str, p95: float, mean: float
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=48, color="#0a9396", alpha=0.9)
    ax.axvline(p95, color="#ae2012", linestyle="--", linewidth=2, label=f"P95={p95:.3f}")
    ax.axvline(mean, color="#005f73", linestyle="-.", linewidth=2, label=f"Mean={mean:.3f}")
    ax.set_xlabel("Wrapped Angular Error (rad)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot harmonic 1-forms and circular coordinates.")
    parser.add_argument("--round-dir", required=True, help="Parity round directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (default: <round-dir>/report/plots)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=350,
        help="Maximum points used for vector-field quiver plots",
    )
    args = parser.parse_args()

    round_dir = Path(args.round_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else round_dir / "report" / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_ambient = load_csv(round_dir / "reference" / "reference_harmonic_ambient.csv")
    cpp_ambient = load_csv(round_dir / "cpp" / "harmonic_ambient.csv")
    ref_circular = load_csv(round_dir / "reference" / "reference_circular_coordinates.csv")
    cpp_circular = load_csv(round_dir / "cpp" / "circular_coordinates.csv")

    n = min(ref_ambient.shape[0], cpp_ambient.shape[0], ref_circular.shape[0], cpp_circular.shape[0])
    ref_ambient = ref_ambient[:n]
    cpp_ambient = cpp_ambient[:n]
    ref_circular = ref_circular[:n]
    cpp_circular = cpp_circular[:n]

    points_ref = np.column_stack([ref_ambient["x"], ref_ambient["y"], ref_ambient["z"]])
    points_cpp = np.column_stack([cpp_ambient["x"], cpp_ambient["y"], cpp_ambient["z"]])

    vec_idx = sample_indices(n, args.max_points)

    ref_forms = ambient_form_indices(ref_ambient.dtype.names)
    cpp_forms = ambient_form_indices(cpp_ambient.dtype.names)
    common_forms = sorted(set(ref_forms).intersection(cpp_forms))

    generated_paths: list[Path] = []
    for form in common_forms[:2]:
        ref_vec = np.column_stack(
            [ref_ambient[f"form{form}_x"], ref_ambient[f"form{form}_y"], ref_ambient[f"form{form}_z"]]
        )
        cpp_vec = np.column_stack(
            [cpp_ambient[f"form{form}_x"], cpp_ambient[f"form{form}_y"], cpp_ambient[f"form{form}_z"]]
        )
        out_path = output_dir / f"harmonic_form_{form}_vector_field.png"
        plot_vector_pair(
            out_path=out_path,
            points_ref=points_ref[vec_idx],
            vec_ref=ref_vec[vec_idx],
            points_cpp=points_cpp[vec_idx],
            vec_cpp=cpp_vec[vec_idx],
            title=f"Harmonic 1-Form {form}: Vector Field",
        )
        generated_paths.append(out_path)

    ref_thetas = theta_indices(ref_circular.dtype.names)
    cpp_thetas = theta_indices(cpp_circular.dtype.names)
    common_thetas = sorted(set(ref_thetas).intersection(cpp_thetas))

    for theta_idx in common_thetas[:2]:
        ref_theta = np.asarray(ref_circular[f"theta_{theta_idx}"], dtype=np.float64)
        cpp_theta = np.asarray(cpp_circular[f"theta_{theta_idx}"], dtype=np.float64)
        pair_path = output_dir / f"circular_theta_{theta_idx}_compare.png"
        plot_angle_pair(
            out_path=pair_path,
            points_ref=points_ref,
            theta_ref=ref_theta,
            points_cpp=points_cpp,
            theta_cpp=cpp_theta,
            title=f"Circular Coordinate theta_{theta_idx}",
        )
        generated_paths.append(pair_path)

        err = wrapped_angle_error(ref_theta, cpp_theta)
        hist_path = output_dir / f"circular_theta_{theta_idx}_error_hist.png"
        plot_angle_error_hist(
            out_path=hist_path,
            errors=err,
            title=f"theta_{theta_idx}: Wrapped Angular Error",
            p95=float(np.percentile(err, 95.0)),
            mean=float(np.mean(err)),
        )
        generated_paths.append(hist_path)

    if len(common_thetas) >= 2:
        theta0 = common_thetas[0]
        theta1 = common_thetas[1]
        grid_path = output_dir / "circular_from_harmonic_forms_compare.png"
        plot_circular_forms_grid(
            out_path=grid_path,
            points_ref=points_ref,
            points_cpp=points_cpp,
            theta_ref0=np.asarray(ref_circular[f"theta_{theta0}"], dtype=np.float64),
            theta_cpp0=np.asarray(cpp_circular[f"theta_{theta0}"], dtype=np.float64),
            theta_ref1=np.asarray(ref_circular[f"theta_{theta1}"], dtype=np.float64),
            theta_cpp1=np.asarray(cpp_circular[f"theta_{theta1}"], dtype=np.float64),
        )
        generated_paths.append(grid_path)

    print(f"plots_dir={output_dir}")
    for p in generated_paths:
        print(f"plot={p}")


if __name__ == "__main__":
    main()
