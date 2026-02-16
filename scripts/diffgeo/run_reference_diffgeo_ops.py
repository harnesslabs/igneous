#!/usr/bin/env python3
import argparse
import csv
import pathlib

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from diffusion_geometry import DiffusionGeometry
from diffusion_geometry.core import build_symmetric_kernel_matrix, knn_graph, markov_chain


def load_points(path: pathlib.Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack([data["x"], data["y"], data["z"]])


def write_table(path: pathlib.Path, header: list[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([float(v) for v in row])


def write_spectrum(path: pathlib.Path, evals: np.ndarray) -> None:
    rows = np.column_stack([np.arange(evals.shape[0], dtype=float), np.asarray(evals, dtype=float)])
    write_table(path, ["mode", "lambda"], rows)


def write_coeffs(path: pathlib.Path, coeff_mat: np.ndarray) -> None:
    rows = np.column_stack([np.arange(coeff_mat.shape[0], dtype=float), coeff_mat])
    header = ["coeff_index"] + [f"form{i}" for i in range(coeff_mat.shape[1])]
    write_table(path, header, rows)


def write_vectors(path: pathlib.Path, points: np.ndarray, vectors: list[np.ndarray], labels: list[str]) -> None:
    cols = [points]
    header = ["x", "y", "z"]
    for label, vec in zip(labels, vectors):
        cols.append(vec)
        header.extend([f"{label}_x", f"{label}_y", f"{label}_z"])
    rows = np.column_stack(cols)
    write_table(path, header, rows)


def compute_deterministic_basis(kernel: np.ndarray, nbr_indices: np.ndarray, n_function_basis: int):
    K_sym, row_sums = build_symmetric_kernel_matrix(kernel, nbr_indices)
    row_sums = np.asarray(row_sums, dtype=float)
    inv_sqrt = np.power(np.maximum(row_sums, 1e-12), -0.5)
    K_norm = diags(inv_sqrt) @ K_sym @ diags(inv_sqrt)

    n = K_norm.shape[0]
    n0 = int(min(max(1, n_function_basis), n))
    if n0 < n:
        _, eigvecs = eigsh(K_norm, n0, which="LM", tol=1e-2, v0=np.ones(n, dtype=float))
    else:
        dense = K_norm.toarray()
        evals, evecs = np.linalg.eigh(dense)
        top_idx = np.argsort(np.abs(evals))[-n0:]
        eigvecs = evecs[:, top_idx[::-1]]

    basis = inv_sqrt[:, None] * eigvecs
    basis = basis[:, ::-1]
    if abs(basis[0, 0]) > 1e-12:
        basis = basis / basis[0, 0]
    measure = row_sums / np.maximum(row_sums.sum(), 1e-12)
    return basis, measure


def select_harmonic_indices(evals: np.ndarray, tol: float, max_modes: int = 3) -> list[int]:
    idx = [int(i) for i, val in enumerate(evals) if abs(float(val)) <= tol]
    idx = idx[:max_modes]
    if not idx:
        idx = list(range(min(max_modes, evals.shape[0])))
    return idx


def dual_from_ambient_2form(ambient: np.ndarray) -> np.ndarray:
    # ambient has shape [n,3,3] antisymmetric. Dual vector = (A12, -A02, A01).
    return np.column_stack([ambient[:, 1, 2], -ambient[:, 0, 2], ambient[:, 0, 1]])


def circular_coordinates(form, lam: float, mode: int):
    dg = form.dg
    operator = form.sharp().operator - lam * dg.laplacian(0)
    evals, efunctions = operator.spectrum()
    circular_indices = np.where(evals.imag > 0)[0]
    if circular_indices.size == 0:
        return np.zeros(dg.n, dtype=float)

    mode_idx = min(max(mode, 0), circular_indices.size - 1)
    circular_funcs = efunctions[circular_indices].to_ambient()
    chosen = circular_funcs[mode_idx]
    return np.mod(np.arctan2(chosen.imag, chosen.real), 2.0 * np.pi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference diffusion geometry ops")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-function-basis", type=int, default=50)
    parser.add_argument("--n-coefficients", type=int, default=50)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--knn-bandwidth", type=int, default=8)
    parser.add_argument("--bandwidth-variability", type=float, default=-0.5)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--harmonic-tolerance", type=float, default=1e-3)
    parser.add_argument("--circular-lambda", type=float, default=1.0)
    parser.add_argument("--circular-mode-0", type=int, default=0)
    parser.add_argument("--circular-mode-1", type=int, default=1)
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    points = load_points(pathlib.Path(args.input_csv))
    nbr_distances, nbr_indices = knn_graph(points, knn_kernel=args.k_neighbors)
    kernel, bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        c=args.c,
        bandwidth_variability=args.bandwidth_variability,
        knn_bandwidth=args.knn_bandwidth,
    )
    function_basis, measure = compute_deterministic_basis(
        kernel=kernel,
        nbr_indices=nbr_indices,
        n_function_basis=args.n_function_basis,
    )

    dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=kernel,
        immersion_coords=None,
        data_matrix=points,
        bandwidths=bandwidths,
        n_function_basis=args.n_function_basis,
        n_coefficients=args.n_coefficients,
        measure=measure,
        function_basis=function_basis,
        use_mean_centres=True,
        regularisation_method="diffusion",
    )

    vals1, forms1 = dg.laplacian(1).spectrum()
    vals2, forms2 = dg.laplacian(2).spectrum()

    harmonic1_idx = select_harmonic_indices(np.asarray(vals1), args.harmonic_tolerance, 3)
    harmonic2_idx = select_harmonic_indices(np.asarray(vals2), args.harmonic_tolerance, 3)

    h1_coeffs = np.column_stack([np.asarray(forms1[i].coeffs, dtype=float) for i in harmonic1_idx])
    h1_ambient = [np.asarray(forms1[i].to_ambient(), dtype=float) for i in harmonic1_idx]

    h2_coeffs = np.column_stack([np.asarray(forms2[i].coeffs, dtype=float) for i in harmonic2_idx])
    h2_ambient = [dual_from_ambient_2form(np.asarray(forms2[i].to_ambient(), dtype=float)) for i in harmonic2_idx]

    idx0 = harmonic1_idx[0]
    idx1 = harmonic1_idx[1] if len(harmonic1_idx) > 1 else harmonic1_idx[0]
    theta0 = circular_coordinates(forms1[idx0], args.circular_lambda, args.circular_mode_0)
    theta1 = circular_coordinates(forms1[idx1], args.circular_lambda, args.circular_mode_1)

    if len(harmonic1_idx) > 1:
        wedge = forms1[harmonic1_idx[0]] ^ forms1[harmonic1_idx[1]]
    else:
        wedge = forms1[harmonic1_idx[0]] ^ forms1[harmonic1_idx[0]]
    wedge_coeffs = np.asarray(wedge.coeffs, dtype=float)[:, None]
    wedge_ambient = dual_from_ambient_2form(np.asarray(wedge.to_ambient(), dtype=float))

    write_table(out / "points.csv", ["x", "y", "z"], points)
    write_spectrum(out / "form1_spectrum.csv", np.asarray(vals1, dtype=float))
    write_spectrum(out / "form2_spectrum.csv", np.asarray(vals2, dtype=float))
    write_coeffs(out / "harmonic1_coeffs.csv", h1_coeffs)
    write_coeffs(out / "harmonic2_coeffs.csv", h2_coeffs)
    write_coeffs(out / "wedge_h1h1_coeffs.csv", wedge_coeffs)
    write_vectors(out / "harmonic1_ambient.csv", points, h1_ambient, [f"form{i}" for i in range(len(h1_ambient))])
    write_vectors(out / "harmonic2_ambient.csv", points, h2_ambient, [f"form{i}" for i in range(len(h2_ambient))])
    write_vectors(out / "wedge_h1h1_ambient.csv", points, [wedge_ambient], ["wedge"])

    circ_rows = np.column_stack([points, theta0, theta1])
    write_table(out / "circular_coordinates.csv", ["x", "y", "z", "theta_0", "theta_1"], circ_rows)


if __name__ == "__main__":
    main()
