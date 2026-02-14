#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        rows: dict[str, list[float]] = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                value = row.get(field, "")
                rows[field].append(float(value) if value not in (None, "") else math.nan)

    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def sorted_prefixed_columns(columns: dict[str, np.ndarray], prefix: str) -> list[str]:
    names = [name for name in columns.keys() if name.startswith(prefix)]

    def sort_key(name: str) -> tuple[int, str]:
        suffix = name[len(prefix) :]
        suffix = suffix.lstrip("_")
        try:
            return (int(suffix), name)
        except ValueError:
            return (10**9, name)

    return sorted(names, key=sort_key)


def to_xyz(columns: dict[str, np.ndarray]) -> np.ndarray:
    required = ("x", "y", "z")
    for field in required:
        if field not in columns:
            raise ValueError(f"Missing column '{field}'")
    return np.column_stack([columns["x"], columns["y"], columns["z"]])


def sorted_ambient_form_indices(columns: dict[str, np.ndarray]) -> list[int]:
    indices: list[int] = []
    for name in columns.keys():
        if not name.startswith("form") or not name.endswith("_x"):
            continue
        middle = name[len("form") : -len("_x")]
        try:
            idx = int(middle)
        except ValueError:
            continue
        if f"form{idx}_y" in columns and f"form{idx}_z" in columns:
            indices.append(idx)
    return sorted(set(indices))


def harmonic_ambient_matrix(
    columns: dict[str, np.ndarray], form_indices: list[int], n_points: int
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for form_idx in form_indices:
        vec = np.column_stack(
            [
                columns[f"form{form_idx}_x"][:n_points],
                columns[f"form{form_idx}_y"][:n_points],
                columns[f"form{form_idx}_z"][:n_points],
            ]
        ).reshape(-1)
        vectors.append(vec)
    return np.column_stack(vectors) if vectors else np.zeros((n_points * 3, 0))


def orthonormal_basis(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64), 0

    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64), 0

    tol = max(rcond * float(s[0]), rcond)
    rank = int(np.sum(s > tol))
    return u[:, :rank], rank


def principal_angles_deg(a: np.ndarray, b: np.ndarray, rcond: float) -> np.ndarray:
    qa, rank_a = orthonormal_basis(a, rcond)
    qb, rank_b = orthonormal_basis(b, rcond)
    rank = min(rank_a, rank_b)
    if rank == 0:
        return np.asarray([90.0], dtype=np.float64)

    cross = qa[:, :rank].T @ qb[:, :rank]
    singular_vals = np.linalg.svd(cross, compute_uv=False)
    singular_vals = np.clip(singular_vals, -1.0, 1.0)
    return np.degrees(np.arccos(singular_vals))


def procrustes_relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size == 0 or candidate.size == 0:
        return float("nan")

    c = candidate.T @ reference
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    r = u @ vt
    aligned = candidate @ r
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    return float(np.linalg.norm(aligned - reference) / denom)


def circular_mode_metrics(theta_ref: np.ndarray, theta_cpp: np.ndarray) -> dict[str, float | str]:
    z_ref = np.exp(1.0j * theta_ref)

    candidates: list[tuple[str, np.ndarray]] = [
        ("direct", np.exp(1.0j * theta_cpp)),
        ("conjugate", np.exp(-1.0j * theta_cpp)),
    ]

    best: dict[str, float | str] | None = None
    for orientation, z_raw in candidates:
        corr_raw = np.vdot(z_raw, z_ref)
        phase = float(np.angle(corr_raw)) if np.abs(corr_raw) > 0 else 0.0
        z_aligned = z_raw * np.exp(1.0j * phase)

        denom = max(float(np.linalg.norm(z_ref) * np.linalg.norm(z_aligned)), 1e-12)
        corr_mag = float(np.abs(np.vdot(z_ref, z_aligned)) / denom)

        angle_diff = np.angle(z_ref * np.conj(z_aligned))
        abs_diff = np.abs(angle_diff)

        entry: dict[str, float | str] = {
            "orientation": orientation,
            "phase_shift_rad": phase,
            "complex_correlation": corr_mag,
            "wrapped_error_mean_rad": float(np.mean(abs_diff)),
            "wrapped_error_p95_rad": float(np.quantile(abs_diff, 0.95)),
            "wrapped_error_max_rad": float(np.max(abs_diff)),
        }

        if best is None:
            best = entry
            continue

        if entry["complex_correlation"] > best["complex_correlation"]:
            best = entry
        elif entry["complex_correlation"] == best["complex_correlation"] and entry[
            "wrapped_error_p95_rad"
        ] < best["wrapped_error_p95_rad"]:
            best = entry

    assert best is not None
    return best


def finite_or_default(value: float, default: float) -> float:
    return value if np.isfinite(value) else default


def composite_score(
    harmonic_subspace_max_angle_deg: float,
    harmonic_procrustes_rel_error: float,
    circular_complex_correlation_mean: float,
    circular_wrapped_p95_rad_max: float,
) -> float:
    # Lower is better.
    angle_component = finite_or_default(harmonic_subspace_max_angle_deg / 90.0, 1.0)
    proc_component = finite_or_default(harmonic_procrustes_rel_error, 1.0)
    corr_component = finite_or_default(1.0 - circular_complex_correlation_mean, 1.0)
    circ_component = finite_or_default(circular_wrapped_p95_rad_max / math.pi, 1.0)

    return float(
        0.35 * angle_component
        + 0.35 * proc_component
        + 0.15 * corr_component
        + 0.15 * circ_component
    )


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    circular = payload["circular_modes"]
    gates = payload["gates"]

    lines: list[str] = []
    lines.append(f"# {payload['label']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Reference dir: `{payload['reference_dir']}`")
    lines.append(f"- C++ dir: `{payload['cpp_dir']}`")
    lines.append("")
    lines.append("## Primary Metrics")
    lines.append("")
    lines.append(f"- Harmonic subspace max principal angle (deg): `{summary['harmonic_subspace_max_angle_deg']:.6f}`")
    lines.append(f"- Harmonic Procrustes relative error: `{summary['harmonic_procrustes_rel_error']:.6f}`")
    lines.append(f"- Circular complex correlation (min): `{summary['circular_complex_correlation_min']:.6f}`")
    lines.append(f"- Circular complex correlation (mean): `{summary['circular_complex_correlation_mean']:.6f}`")
    lines.append(f"- Circular wrapped angular P95 error max (rad): `{summary['circular_wrapped_p95_rad_max']:.6f}`")
    lines.append(f"- Composite parity score (lower is better): `{summary['composite_score']:.6f}`")
    lines.append("")
    lines.append("## Circular Modes")
    lines.append("")
    lines.append("| Mode | Orientation | Correlation | P95 (rad) | Mean (rad) | Max (rad) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in circular:
        lines.append(
            f"| `{row['name']}` | `{row['orientation']}` | {row['complex_correlation']:.6f} | {row['wrapped_error_p95_rad']:.6f} | {row['wrapped_error_mean_rad']:.6f} | {row['wrapped_error_max_rad']:.6f} |"
        )

    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append(f"- Final pass gate: `{'pass' if gates['final_pass'] else 'fail'}`")
    lines.append(f"- Commit improvement gate: `{'pass' if gates['commit_gate_pass'] else 'fail'}`")
    lines.append(f"- Commit gate details: `{gates['commit_gate_details']}`")
    if gates.get("improvement_vs_previous") is not None:
        lines.append(
            f"- Composite improvement vs previous: `{100.0 * gates['improvement_vs_previous']:.2f}%`"
        )

    if payload["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reference and C++ Hodge outputs")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--cpp-dir", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", default="Hodge parity comparison")
    parser.add_argument("--previous-json", default="")
    parser.add_argument("--gate-improvement", type=float, default=0.20)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--final-max-angle-deg", type=float, default=3.0)
    parser.add_argument("--final-max-procrustes", type=float, default=0.05)
    parser.add_argument("--final-min-correlation", type=float, default=0.99)
    parser.add_argument("--final-max-p95-rad", type=float, default=0.10)
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    cpp_dir = Path(args.cpp_dir)
    output_markdown = Path(args.output_markdown)
    output_json = Path(args.output_json)

    warnings: list[str] = []

    ref_points_cols = load_csv_columns(reference_dir / "reference_points.csv")
    cpp_points_cols = load_csv_columns(cpp_dir / "points.csv")
    ref_points = to_xyz(ref_points_cols)
    cpp_points = to_xyz(cpp_points_cols)

    n_points = min(ref_points.shape[0], cpp_points.shape[0])
    if ref_points.shape[0] != cpp_points.shape[0]:
        warnings.append(
            f"Point count mismatch: reference={ref_points.shape[0]}, cpp={cpp_points.shape[0]}; cropped to {n_points}."
        )
    ref_points = ref_points[:n_points]
    cpp_points = cpp_points[:n_points]

    point_diff = ref_points - cpp_points
    point_rmse = float(np.sqrt(np.mean(point_diff**2)))
    point_max_abs = float(np.max(np.abs(point_diff)))
    if point_rmse > 1e-6:
        warnings.append(f"Input point clouds differ (rmse={point_rmse:.3e}).")

    ref_ambient_cols = load_csv_columns(reference_dir / "reference_harmonic_ambient.csv")
    cpp_ambient_cols = load_csv_columns(cpp_dir / "harmonic_ambient.csv")
    ref_form_indices = sorted_ambient_form_indices(ref_ambient_cols)
    cpp_form_indices = sorted_ambient_form_indices(cpp_ambient_cols)
    common_form_indices = sorted(set(ref_form_indices).intersection(cpp_form_indices))
    if len(ref_form_indices) != len(cpp_form_indices):
        warnings.append(
            f"Harmonic ambient form count mismatch: reference={len(ref_form_indices)}, cpp={len(cpp_form_indices)}."
        )
    if not common_form_indices:
        raise ValueError("No common harmonic ambient form columns found.")

    ref_harmonic_ambient = harmonic_ambient_matrix(
        ref_ambient_cols, common_form_indices, n_points
    )
    cpp_harmonic_ambient = harmonic_ambient_matrix(
        cpp_ambient_cols, common_form_indices, n_points
    )

    angles_deg = principal_angles_deg(ref_harmonic_ambient, cpp_harmonic_ambient, args.rcond)
    max_angle_deg = float(np.max(angles_deg))
    procrustes_err = procrustes_relative_error(ref_harmonic_ambient, cpp_harmonic_ambient)

    # Secondary diagnostics in coefficient space (basis-order dependent).
    ref_coeff_cols = load_csv_columns(reference_dir / "reference_harmonic_coeffs.csv")
    cpp_coeff_cols = load_csv_columns(cpp_dir / "harmonic_coeffs.csv")
    ref_form_cols = sorted_prefixed_columns(ref_coeff_cols, "form")
    cpp_form_cols = sorted_prefixed_columns(cpp_coeff_cols, "form")
    n_coeff_forms = min(len(ref_form_cols), len(cpp_form_cols))
    if n_coeff_forms == 0:
        raise ValueError("No common harmonic coefficient form columns found.")

    ref_coeff = np.column_stack([ref_coeff_cols[name] for name in ref_form_cols[:n_coeff_forms]])
    cpp_coeff = np.column_stack([cpp_coeff_cols[name] for name in cpp_form_cols[:n_coeff_forms]])
    n_coeff_rows = min(ref_coeff.shape[0], cpp_coeff.shape[0])
    if ref_coeff.shape[0] != cpp_coeff.shape[0]:
        warnings.append(
            f"Harmonic coefficient row mismatch: reference={ref_coeff.shape[0]}, cpp={cpp_coeff.shape[0]}; cropped to {n_coeff_rows}."
        )
    ref_coeff = ref_coeff[:n_coeff_rows]
    cpp_coeff = cpp_coeff[:n_coeff_rows]
    coeff_angles_deg = principal_angles_deg(ref_coeff, cpp_coeff, args.rcond)
    coeff_procrustes_err = procrustes_relative_error(ref_coeff, cpp_coeff)

    ref_circ_cols = load_csv_columns(reference_dir / "reference_circular_coordinates.csv")
    cpp_circ_cols = load_csv_columns(cpp_dir / "circular_coordinates.csv")

    ref_theta_cols = sorted_prefixed_columns(ref_circ_cols, "theta")
    cpp_theta_cols = sorted_prefixed_columns(cpp_circ_cols, "theta")
    n_theta = min(len(ref_theta_cols), len(cpp_theta_cols))
    if n_theta == 0:
        raise ValueError("No common circular coordinate columns found.")

    circular_mode_rows: list[dict[str, Any]] = []
    circular_corr_values: list[float] = []
    circular_p95_values: list[float] = []

    for idx in range(n_theta):
        ref_col = ref_theta_cols[idx]
        cpp_col = cpp_theta_cols[idx]

        theta_ref = ref_circ_cols[ref_col][:n_points]
        theta_cpp = cpp_circ_cols[cpp_col][:n_points]
        metrics = circular_mode_metrics(theta_ref, theta_cpp)

        row = {
            "name": f"theta_{idx}",
            "reference_column": ref_col,
            "cpp_column": cpp_col,
            **metrics,
        }
        circular_mode_rows.append(row)
        circular_corr_values.append(float(metrics["complex_correlation"]))
        circular_p95_values.append(float(metrics["wrapped_error_p95_rad"]))

    circular_corr_min = float(np.min(circular_corr_values))
    circular_corr_mean = safe_mean(circular_corr_values)
    circular_p95_max = float(np.max(circular_p95_values))

    score = composite_score(
        harmonic_subspace_max_angle_deg=max_angle_deg,
        harmonic_procrustes_rel_error=procrustes_err,
        circular_complex_correlation_mean=circular_corr_mean,
        circular_wrapped_p95_rad_max=circular_p95_max,
    )

    final_pass = (
        max_angle_deg <= args.final_max_angle_deg
        and procrustes_err <= args.final_max_procrustes
        and circular_corr_min >= args.final_min_correlation
        and circular_p95_max <= args.final_max_p95_rad
    )

    improvement_vs_previous: float | None = None
    commit_gate_pass = True
    commit_gate_details = "baseline run (no previous composite score provided)"

    if args.previous_json:
        previous_path = Path(args.previous_json)
        if previous_path.exists():
            previous_payload = json.loads(previous_path.read_text())
            previous_score = float(previous_payload.get("summary", {}).get("composite_score", "nan"))
            if np.isfinite(previous_score) and previous_score > 0.0:
                improvement_vs_previous = (previous_score - score) / previous_score
                commit_gate_pass = improvement_vs_previous >= args.gate_improvement
                commit_gate_details = (
                    f"improvement={100.0 * improvement_vs_previous:.2f}% (threshold={100.0 * args.gate_improvement:.2f}%)"
                )
            else:
                commit_gate_pass = False
                commit_gate_details = "previous composite score missing or non-positive"
        else:
            commit_gate_pass = False
            commit_gate_details = f"previous report not found: {previous_path}"

    payload: dict[str, Any] = {
        "label": args.label,
        "reference_dir": str(reference_dir),
        "cpp_dir": str(cpp_dir),
        "summary": {
            "harmonic_subspace_max_angle_deg": max_angle_deg,
            "harmonic_procrustes_rel_error": procrustes_err,
            "circular_complex_correlation_min": circular_corr_min,
            "circular_complex_correlation_mean": circular_corr_mean,
            "circular_wrapped_p95_rad_max": circular_p95_max,
            "composite_score": score,
        },
        "harmonic_details": {
            "n_forms_compared": len(common_form_indices),
            "form_indices_compared": common_form_indices,
            "ambient_principal_angles_deg": [float(v) for v in angles_deg.tolist()],
            "ambient_procrustes_rel_error": procrustes_err,
            "coeff_principal_angles_deg": [float(v) for v in coeff_angles_deg.tolist()],
            "coeff_procrustes_rel_error": coeff_procrustes_err,
            "n_coeff_rows_compared": n_coeff_rows,
        },
        "circular_modes": circular_mode_rows,
        "point_cloud": {
            "n_points_compared": n_points,
            "point_rmse": point_rmse,
            "point_max_abs": point_max_abs,
        },
        "gates": {
            "final_pass": final_pass,
            "commit_gate_pass": commit_gate_pass,
            "commit_gate_details": commit_gate_details,
            "improvement_vs_previous": improvement_vs_previous,
            "thresholds": {
                "final_max_angle_deg": args.final_max_angle_deg,
                "final_max_procrustes": args.final_max_procrustes,
                "final_min_correlation": args.final_min_correlation,
                "final_max_p95_rad": args.final_max_p95_rad,
                "commit_improvement": args.gate_improvement,
            },
        },
        "warnings": warnings,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    output_markdown.write_text(render_markdown(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
