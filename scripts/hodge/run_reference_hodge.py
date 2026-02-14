#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
from typing import List

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from diffusion_geometry import DiffusionGeometry
from diffusion_geometry.core import (
    build_symmetric_kernel_matrix,
    knn_graph,
    markov_chain,
)


def load_points(path: pathlib.Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack([data["x"], data["y"], data["z"]])


def write_table(path: pathlib.Path, header: List[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([float(v) for v in row])


def compute_deterministic_basis(
    kernel: np.ndarray, nbr_indices: np.ndarray, n_function_basis: int
) -> tuple[np.ndarray, np.ndarray]:
    K_sym, row_sums = build_symmetric_kernel_matrix(kernel, nbr_indices)
    row_sums = np.asarray(row_sums, dtype=float)
    inv_sqrt = np.power(np.maximum(row_sums, 1e-12), -0.5)
    K_normalized = diags(inv_sqrt) @ K_sym @ diags(inv_sqrt)

    n = K_normalized.shape[0]
    n0 = int(min(max(1, n_function_basis), n))

    if n0 < n:
        _, eigenfunctions = eigsh(
            K_normalized,
            n0,
            which="LM",
            tol=1e-2,
            v0=np.ones(n, dtype=float),
        )
    else:
        dense = K_normalized.toarray()
        evals, evecs = np.linalg.eigh(dense)
        top_idx = np.argsort(np.abs(evals))[-n0:]
        eigenfunctions = evecs[:, top_idx[::-1]]

    basis = inv_sqrt[:, None] * eigenfunctions
    basis = basis[:, ::-1]
    if abs(basis[0, 0]) > 1e-12:
        basis = basis / basis[0, 0]

    measure = row_sums / np.maximum(row_sums.sum(), 1e-12)
    return basis, measure


def circular_coordinates(form, lam: float, mode: int):
    dg = form.dg
    operator = form.sharp().operator - lam * dg.laplacian(0)
    evals, efunctions = operator.spectrum()

    circular_indices = np.where(evals.imag > 0)[0]
    if circular_indices.size == 0:
        zeros = np.zeros(dg.n, dtype=float)
        return zeros, complex(0.0, 0.0), -1

    mode_idx = min(max(mode, 0), circular_indices.size - 1)
    chosen_global = int(circular_indices[mode_idx])
    chosen_eval = evals[chosen_global]

    circular_funcs = efunctions[circular_indices].to_ambient()
    chosen_func = circular_funcs[mode_idx]
    angles = np.mod(np.arctan2(chosen_func.imag, chosen_func.real), 2.0 * np.pi)
    return angles, chosen_eval, chosen_global


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference DiffusionGeometry Hodge pipeline")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-function-basis", type=int, default=50)
    parser.add_argument("--n-coefficients", type=int, default=50)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--knn-bandwidth", type=int, default=8)
    parser.add_argument("--bandwidth-variability", type=float, default=-0.5)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--circular-lambda", type=float, default=1.0)
    parser.add_argument("--form-indices", default="0,1")
    parser.add_argument("--mode-indices", default="0,0")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        immersion_coords=points,
        data_matrix=points,
        bandwidths=bandwidths,
        n_function_basis=args.n_function_basis,
        n_coefficients=args.n_coefficients,
        measure=measure,
        function_basis=function_basis,
        use_mean_centres=True,
        regularisation_method="diffusion",
    )

    vals_1, forms_1 = dg.laplacian(1).spectrum()

    form_indices = [int(x.strip()) for x in args.form_indices.split(",") if x.strip()]
    mode_indices = [int(x.strip()) for x in args.mode_indices.split(",") if x.strip()]
    if len(mode_indices) < len(form_indices):
        mode_indices.extend([0] * (len(form_indices) - len(mode_indices)))

    harmonic_coeffs = []
    harmonic_ambient = []
    circular_thetas = []
    circular_meta = []

    for idx, form_idx in enumerate(form_indices):
        form = forms_1[form_idx]
        harmonic_coeffs.append(np.asarray(form.coeffs, dtype=float))
        ambient = np.asarray(form.to_ambient(), dtype=float)
        harmonic_ambient.append(ambient)

        theta, eigval, eig_idx = circular_coordinates(
            form, args.circular_lambda, mode_indices[idx]
        )
        circular_thetas.append(theta)
        circular_meta.append(
            {
                "name": f"theta_{idx}",
                "form_index": int(form_idx),
                "mode": int(mode_indices[idx]),
                "selected_global_index": int(eig_idx),
                "lambda": float(args.circular_lambda),
                "eigenvalue_real": float(np.real(eigval)),
                "eigenvalue_imag": float(np.imag(eigval)),
            }
        )

    write_table(
        out_dir / "reference_points.csv",
        ["x", "y", "z"],
        points,
    )

    spectrum_rows = np.column_stack([np.arange(vals_1.shape[0]), np.asarray(vals_1, dtype=float)])
    write_table(
        out_dir / "reference_hodge_spectrum.csv",
        ["mode", "lambda"],
        spectrum_rows,
    )

    coeff_mat = np.column_stack(harmonic_coeffs)
    coeff_rows = np.column_stack([np.arange(coeff_mat.shape[0]), coeff_mat])
    coeff_header = ["coeff_index"] + [f"form{i}" for i in range(coeff_mat.shape[1])]
    write_table(out_dir / "reference_harmonic_coeffs.csv", coeff_header, coeff_rows)

    ambient_cols = [points]
    ambient_header = ["x", "y", "z"]
    for i, ambient in enumerate(harmonic_ambient):
        ambient_cols.append(ambient)
        ambient_header.extend([f"form{i}_x", f"form{i}_y", f"form{i}_z"])
    ambient_rows = np.column_stack(ambient_cols)
    write_table(out_dir / "reference_harmonic_ambient.csv", ambient_header, ambient_rows)

    circular_rows = np.column_stack([points] + circular_thetas)
    circular_header = ["x", "y", "z"] + [f"theta_{i}" for i in range(len(circular_thetas))]
    write_table(out_dir / "reference_circular_coordinates.csv", circular_header, circular_rows)

    with (out_dir / "reference_circular_modes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "form_index", "mode", "selected_global_index", "lambda", "eigenvalue_real", "eigenvalue_imag"])
        for row in circular_meta:
            writer.writerow(
                [
                    row["name"],
                    row["form_index"],
                    row["mode"],
                    row["selected_global_index"],
                    row["lambda"],
                    row["eigenvalue_real"],
                    row["eigenvalue_imag"],
                ]
            )

    metadata = {
        "n_points": int(points.shape[0]),
        "n_function_basis": int(args.n_function_basis),
        "n_coefficients": int(args.n_coefficients),
        "k_neighbors": int(args.k_neighbors),
        "knn_bandwidth": int(args.knn_bandwidth),
        "bandwidth_variability": float(args.bandwidth_variability),
        "c": float(args.c),
        "circular_lambda": float(args.circular_lambda),
        "form_indices": form_indices,
        "mode_indices": mode_indices,
    }
    (out_dir / "reference_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
