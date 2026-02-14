#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.genfromtxt(path, delimiter=",", names=True)


def set_equal_axes(ax, x, y, z):
    mins = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    maxs = np.array([x.max(), y.max(), z.max()], dtype=np.float64)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    step = max(1, n // max_points)
    return np.arange(0, n, step, dtype=np.int64)[:max_points]


def plot_vector_pair(out_path: Path, points_ref, vec_ref, points_cpp, vec_cpp, title: str):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, pts, vecs, label in [(ax1, points_ref, vec_ref, "Reference"), (ax2, points_cpp, vec_cpp, "C++")]:
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
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_angle_pair(out_path: Path, points_ref, theta_ref, points_cpp, theta_cpp, title: str):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, pts, theta, label in [(ax1, points_ref, theta_ref, "Reference"), (ax2, points_cpp, theta_cpp, "C++")]:
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
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.03)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def wrapped_angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diffusion topology parity outputs")
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-points", type=int, default=350)
    args = parser.parse_args()

    round_dir = Path(args.round_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (round_dir / "report" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_torus = load_csv(round_dir / "reference" / "torus" / "harmonic1_ambient.csv")
    cpp_torus = load_csv(round_dir / "cpp" / "torus" / "harmonic1_ambient.csv")
    ref_torus_circ = load_csv(round_dir / "reference" / "torus" / "circular_coordinates.csv")
    cpp_torus_circ = load_csv(round_dir / "cpp" / "torus" / "circular_coordinates.csv")

    ref_sphere_h2 = load_csv(round_dir / "reference" / "sphere" / "harmonic2_ambient.csv")
    cpp_sphere_h2 = load_csv(round_dir / "cpp" / "sphere" / "harmonic2_ambient.csv")

    ref_wedge = load_csv(round_dir / "reference" / "torus" / "wedge_h1h1_ambient.csv")
    cpp_wedge = load_csv(round_dir / "cpp" / "torus" / "wedge_h1h1_ambient.csv")

    n_torus = min(ref_torus.shape[0], cpp_torus.shape[0])
    n_sphere = min(ref_sphere_h2.shape[0], cpp_sphere_h2.shape[0])

    pts_ref_torus = np.column_stack([ref_torus["x"][:n_torus], ref_torus["y"][:n_torus], ref_torus["z"][:n_torus]])
    pts_cpp_torus = np.column_stack([cpp_torus["x"][:n_torus], cpp_torus["y"][:n_torus], cpp_torus["z"][:n_torus]])
    pts_ref_sphere = np.column_stack([ref_sphere_h2["x"][:n_sphere], ref_sphere_h2["y"][:n_sphere], ref_sphere_h2["z"][:n_sphere]])
    pts_cpp_sphere = np.column_stack([cpp_sphere_h2["x"][:n_sphere], cpp_sphere_h2["y"][:n_sphere], cpp_sphere_h2["z"][:n_sphere]])

    idx_torus = sample_indices(n_torus, args.max_points)
    idx_sphere = sample_indices(n_sphere, args.max_points)

    generated: list[str] = []

    for form_idx in [0, 1]:
        keyx = f"form{form_idx}_x"
        keyy = f"form{form_idx}_y"
        keyz = f"form{form_idx}_z"
        if keyx not in ref_torus.dtype.names or keyx not in cpp_torus.dtype.names:
            continue
        ref_vec = np.column_stack([ref_torus[keyx], ref_torus[keyy], ref_torus[keyz]])
        cpp_vec = np.column_stack([cpp_torus[keyx], cpp_torus[keyy], cpp_torus[keyz]])
        out = out_dir / f"harmonic1_form{form_idx}_vector_compare.png"
        plot_vector_pair(out, pts_ref_torus[idx_torus], ref_vec[idx_torus], pts_cpp_torus[idx_torus], cpp_vec[idx_torus], f"Harmonic 1-Form {form_idx} (Torus)")
        generated.append(out.name)

    # Harmonic 2 compare on sphere (form0)
    if "form0_x" in ref_sphere_h2.dtype.names and "form0_x" in cpp_sphere_h2.dtype.names:
        ref_vec2 = np.column_stack([ref_sphere_h2["form0_x"], ref_sphere_h2["form0_y"], ref_sphere_h2["form0_z"]])
        cpp_vec2 = np.column_stack([cpp_sphere_h2["form0_x"], cpp_sphere_h2["form0_y"], cpp_sphere_h2["form0_z"]])
        out = out_dir / "harmonic2_compare.png"
        plot_vector_pair(out, pts_ref_sphere[idx_sphere], ref_vec2[idx_sphere], pts_cpp_sphere[idx_sphere], cpp_vec2[idx_sphere], "Harmonic 2-Form (Dual Vector, Sphere)")
        generated.append(out.name)

    # Wedge vs harmonic2 on torus (reference and C++)
    if "wedge_x" in ref_wedge.dtype.names and "form0_x" in ref_torus.dtype.names:
        fig = plt.figure(figsize=(14, 10))
        panels = [
            (fig.add_subplot(2, 2, 1, projection="3d"), pts_ref_torus[idx_torus], np.column_stack([ref_wedge["wedge_x"], ref_wedge["wedge_y"], ref_wedge["wedge_z"]])[idx_torus], "Reference Wedge(h1,h1)"),
            (fig.add_subplot(2, 2, 2, projection="3d"), pts_cpp_torus[idx_torus], np.column_stack([cpp_wedge["wedge_x"], cpp_wedge["wedge_y"], cpp_wedge["wedge_z"]])[idx_torus], "C++ Wedge(h1,h1)"),
            (fig.add_subplot(2, 2, 3, projection="3d"), pts_ref_torus[idx_torus], np.column_stack([ref_torus["form0_x"], ref_torus["form0_y"], ref_torus["form0_z"]])[idx_torus], "Reference Harmonic2 form0"),
            (fig.add_subplot(2, 2, 4, projection="3d"), pts_cpp_torus[idx_torus], np.column_stack([cpp_torus["form0_x"], cpp_torus["form0_y"], cpp_torus["form0_z"]])[idx_torus], "C++ Harmonic2 form0"),
        ]
        for ax, pts, vecs, title in panels:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c="lightgray", alpha=0.35)
            ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], vecs[:, 0], vecs[:, 1], vecs[:, 2], length=0.18, normalize=True, linewidth=0.5, color="#0a9396")
            set_equal_axes(ax, pts[:, 0], pts[:, 1], pts[:, 2])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        fig.suptitle("Wedge(h1,h1) vs Harmonic 2-Form (Torus)")
        fig.tight_layout()
        out = out_dir / "wedge_h1h1_vs_h2_compare.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        generated.append(out.name)

    # form2 spectrum compare (torus + sphere)
    ref_spec_t = load_csv(round_dir / "reference" / "torus" / "form2_spectrum.csv")
    cpp_spec_t = load_csv(round_dir / "cpp" / "torus" / "form2_spectrum.csv")
    ref_spec_s = load_csv(round_dir / "reference" / "sphere" / "form2_spectrum.csv")
    cpp_spec_s = load_csv(round_dir / "cpp" / "sphere" / "form2_spectrum.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, arr, style in [
        ("Ref Torus", ref_spec_t, "-"),
        ("C++ Torus", cpp_spec_t, "--"),
        ("Ref Sphere", ref_spec_s, "-"),
        ("C++ Sphere", cpp_spec_s, "--"),
    ]:
        m = min(30, arr.shape[0])
        ax.plot(np.arange(m), arr["lambda"][:m], style, label=lbl)
    ax.set_title("Form-2 Spectrum Comparison")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Lambda")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "form2_spectrum_compare.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    generated.append(out.name)

    # Circular theta plots (torus)
    for i in [0, 1]:
        theta = f"theta_{i}"
        if theta not in ref_torus_circ.dtype.names or theta not in cpp_torus_circ.dtype.names:
            continue
        out = out_dir / f"circular_theta{i}_compare.png"
        plot_angle_pair(out, pts_ref_torus, ref_torus_circ[theta][:n_torus], pts_cpp_torus, cpp_torus_circ[theta][:n_torus], f"Circular Coordinate {theta} (Torus)")
        generated.append(out.name)

    # Error histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    e0 = wrapped_angle_error(ref_torus_circ["theta_0"][:n_torus], cpp_torus_circ["theta_0"][:n_torus])
    e1 = wrapped_angle_error(ref_torus_circ["theta_1"][:n_torus], cpp_torus_circ["theta_1"][:n_torus])
    ew = np.linalg.norm(
        np.column_stack([ref_wedge["wedge_x"][:n_torus], ref_wedge["wedge_y"][:n_torus], ref_wedge["wedge_z"][:n_torus]])
        - np.column_stack([cpp_wedge["wedge_x"][:n_torus], cpp_wedge["wedge_y"][:n_torus], cpp_wedge["wedge_z"][:n_torus]]),
        axis=1,
    )
    axes[0].hist(e0, bins=45, color="#0a9396")
    axes[0].set_title("theta_0 wrapped error")
    axes[1].hist(e1, bins=45, color="#0a9396")
    axes[1].set_title("theta_1 wrapped error")
    axes[2].hist(ew, bins=45, color="#bb3e03")
    axes[2].set_title("wedge vector L2 error")
    for ax in axes:
        ax.set_ylabel("count")
    fig.tight_layout()
    out = out_dir / "error_histograms.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    generated.append(out.name)

    index_md = out_dir / "plots_index.md"
    lines = ["# Diffusion Topology Plot Index", ""]
    lines += [f"- {name}" for name in generated]
    index_md.write_text("\n".join(lines) + "\n")

    print(f"plots_dir={out_dir}")
    for name in generated:
        print(f"plot={out_dir / name}")
    print(f"index={index_md}")


if __name__ == "__main__":
    main()
