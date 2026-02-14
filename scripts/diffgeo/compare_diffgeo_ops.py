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
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in reader.fieldnames:
                v = row.get(k, "")
                rows[k].append(float(v) if v not in (None, "") else math.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def sorted_form_indices(cols: dict[str, np.ndarray]) -> list[int]:
    idx: list[int] = []
    for name in cols:
        if not name.startswith("form") or not name.endswith("_x"):
            continue
        mid = name[len("form") : -len("_x")]
        try:
            i = int(mid)
        except ValueError:
            continue
        if f"form{i}_y" in cols and f"form{i}_z" in cols:
            idx.append(i)
    return sorted(set(idx))


def form_matrix(cols: dict[str, np.ndarray], form_indices: list[int], n_points: int) -> np.ndarray:
    mats: list[np.ndarray] = []
    for idx in form_indices:
        vec = np.column_stack([
            cols[f"form{idx}_x"][:n_points],
            cols[f"form{idx}_y"][:n_points],
            cols[f"form{idx}_z"][:n_points],
        ]).reshape(-1)
        mats.append(vec)
    if not mats:
        return np.zeros((n_points * 3, 0), dtype=np.float64)
    return np.column_stack(mats)


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
    qa, ra = orthonormal_basis(a, rcond)
    qb, rb = orthonormal_basis(b, rcond)
    rank = min(ra, rb)
    if rank == 0:
        return np.asarray([90.0], dtype=np.float64)
    cross = qa[:, :rank].T @ qb[:, :rank]
    s = np.linalg.svd(cross, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


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
    candidates = [
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

        row: dict[str, float | str] = {
            "orientation": orientation,
            "complex_correlation": corr_mag,
            "wrapped_error_mean_rad": float(np.mean(abs_diff)),
            "wrapped_error_p95_rad": float(np.quantile(abs_diff, 0.95)),
            "wrapped_error_max_rad": float(np.max(abs_diff)),
            "phase_shift_rad": phase,
        }

        if best is None:
            best = row
        elif row["complex_correlation"] > best["complex_correlation"]:
            best = row
        elif row["complex_correlation"] == best["complex_correlation"] and row[
            "wrapped_error_p95_rad"
        ] < best["wrapped_error_p95_rad"]:
            best = row

    assert best is not None
    return best


def wedge_relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    denom = max(float(np.linalg.norm(ref)), 1e-12)
    e1 = float(np.linalg.norm(ref - cand) / denom)
    e2 = float(np.linalg.norm(ref + cand) / denom)
    return min(e1, e2)


def spectrum_relative_error(reference: np.ndarray, candidate: np.ndarray, count: int = 20) -> float:
    n = min(count, reference.shape[0], candidate.shape[0])
    if n == 0:
        return float("nan")
    ref = reference[:n]
    cand = candidate[:n]
    denom = max(float(np.linalg.norm(ref)), 1e-12)
    return float(np.linalg.norm(ref - cand) / denom)


def finite_or_default(v: float, default: float) -> float:
    return v if np.isfinite(v) else default


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def composite_score(summary: dict[str, float]) -> float:
    h1_angle = finite_or_default(summary["harmonic1_subspace_max_angle_deg"] / 90.0, 1.0)
    h1_proc = finite_or_default(summary["harmonic1_procrustes_rel_error"], 1.0)
    h2_angle = finite_or_default(summary["harmonic2_subspace_max_angle_deg"] / 90.0, 1.0)
    h2_proc = finite_or_default(summary["harmonic2_procrustes_rel_error"], 1.0)
    wedge = finite_or_default(summary["wedge_rel_error"], 1.0)
    spec2 = finite_or_default(summary["form2_spectrum_rel_error"], 1.0)
    corr_penalty = finite_or_default(1.0 - summary["circular_complex_correlation_min"], 1.0)
    circ_p95 = finite_or_default(summary["circular_wrapped_p95_rad_max"] / math.pi, 1.0)

    return float(
        0.20 * h1_angle
        + 0.20 * h1_proc
        + 0.15 * h2_angle
        + 0.15 * h2_proc
        + 0.10 * wedge
        + 0.10 * spec2
        + 0.05 * corr_penalty
        + 0.05 * circ_p95
    )


def compare_dataset(ref_dir: Path, cpp_dir: Path, rcond: float) -> dict[str, Any]:
    warnings: list[str] = []

    ref_points = load_csv_columns(ref_dir / "points.csv")
    cpp_points = load_csv_columns(cpp_dir / "points.csv")
    n_points = min(ref_points["x"].shape[0], cpp_points["x"].shape[0])

    if ref_points["x"].shape[0] != cpp_points["x"].shape[0]:
        warnings.append(
            f"point count mismatch: ref={ref_points['x'].shape[0]} cpp={cpp_points['x'].shape[0]}"
        )

    ref_h1 = load_csv_columns(ref_dir / "harmonic1_ambient.csv")
    cpp_h1 = load_csv_columns(cpp_dir / "harmonic1_ambient.csv")
    ref_h1_idx = sorted_form_indices(ref_h1)
    cpp_h1_idx = sorted_form_indices(cpp_h1)
    common_h1_idx = sorted(set(ref_h1_idx).intersection(cpp_h1_idx))
    if not common_h1_idx:
        raise ValueError(f"No common harmonic1 forms in {ref_dir}")

    ref_h1_mat = form_matrix(ref_h1, common_h1_idx, n_points)
    cpp_h1_mat = form_matrix(cpp_h1, common_h1_idx, n_points)

    ref_h2 = load_csv_columns(ref_dir / "harmonic2_ambient.csv")
    cpp_h2 = load_csv_columns(cpp_dir / "harmonic2_ambient.csv")
    ref_h2_idx = sorted_form_indices(ref_h2)
    cpp_h2_idx = sorted_form_indices(cpp_h2)
    common_h2_idx = sorted(set(ref_h2_idx).intersection(cpp_h2_idx))
    if not common_h2_idx:
        raise ValueError(f"No common harmonic2 forms in {ref_dir}")

    ref_h2_mat = form_matrix(ref_h2, common_h2_idx, n_points)
    cpp_h2_mat = form_matrix(cpp_h2, common_h2_idx, n_points)

    h1_angles = principal_angles_deg(ref_h1_mat, cpp_h1_mat, rcond)
    h2_angles = principal_angles_deg(ref_h2_mat, cpp_h2_mat, rcond)

    h1_proc = procrustes_relative_error(ref_h1_mat, cpp_h1_mat)
    h2_proc = procrustes_relative_error(ref_h2_mat, cpp_h2_mat)

    ref_wedge = load_csv_columns(ref_dir / "wedge_h1h1_ambient.csv")
    cpp_wedge = load_csv_columns(cpp_dir / "wedge_h1h1_ambient.csv")
    ref_wedge_vec = np.column_stack(
        [ref_wedge["wedge_x"][:n_points], ref_wedge["wedge_y"][:n_points], ref_wedge["wedge_z"][:n_points]]
    )
    cpp_wedge_vec = np.column_stack(
        [cpp_wedge["wedge_x"][:n_points], cpp_wedge["wedge_y"][:n_points], cpp_wedge["wedge_z"][:n_points]]
    )
    wedge_err = wedge_relative_error(ref_wedge_vec, cpp_wedge_vec)

    ref_spec2 = load_csv_columns(ref_dir / "form2_spectrum.csv")
    cpp_spec2 = load_csv_columns(cpp_dir / "form2_spectrum.csv")
    spec2_err = spectrum_relative_error(ref_spec2["lambda"], cpp_spec2["lambda"], 20)

    ref_circ = load_csv_columns(ref_dir / "circular_coordinates.csv")
    cpp_circ = load_csv_columns(cpp_dir / "circular_coordinates.csv")
    circular_rows = []
    circ_corr = []
    circ_p95 = []
    for idx in [0, 1]:
        name = f"theta_{idx}"
        if name not in ref_circ or name not in cpp_circ:
            warnings.append(f"missing circular column {name}")
            continue
        metrics = circular_mode_metrics(ref_circ[name][:n_points], cpp_circ[name][:n_points])
        circular_rows.append({"name": name, **metrics})
        circ_corr.append(float(metrics["complex_correlation"]))
        circ_p95.append(float(metrics["wrapped_error_p95_rad"]))

    return {
        "warnings": warnings,
        "harmonic1_subspace_max_angle_deg": float(np.max(h1_angles)),
        "harmonic1_procrustes_rel_error": h1_proc,
        "harmonic2_subspace_max_angle_deg": float(np.max(h2_angles)),
        "harmonic2_procrustes_rel_error": h2_proc,
        "wedge_rel_error": wedge_err,
        "form2_spectrum_rel_error": spec2_err,
        "circular_modes": circular_rows,
        "circular_complex_correlation_min": min(circ_corr) if circ_corr else float("nan"),
        "circular_complex_correlation_mean": safe_mean(circ_corr),
        "circular_wrapped_p95_rad_max": max(circ_p95) if circ_p95 else float("nan"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {payload['label']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in payload["summary"].items():
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Dataset Metrics")
    lines.append("")
    for name, metrics in payload["datasets"].items():
        lines.append(f"### {name}")
        for key in [
            "harmonic1_subspace_max_angle_deg",
            "harmonic1_procrustes_rel_error",
            "harmonic2_subspace_max_angle_deg",
            "harmonic2_procrustes_rel_error",
            "wedge_rel_error",
            "form2_spectrum_rel_error",
            "circular_complex_correlation_min",
            "circular_wrapped_p95_rad_max",
        ]:
            val = metrics[key]
            lines.append(f"- {key}: `{val:.6f}`")
        if metrics["warnings"]:
            lines.append("- warnings:")
            for w in metrics["warnings"]:
                lines.append(f"  - {w}")
        lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append(f"- final_pass: `{'pass' if payload['gates']['final_pass'] else 'fail'}`")
    lines.append(
        f"- commit_gate_pass: `{'pass' if payload['gates']['commit_gate_pass'] else 'fail'}`"
    )
    lines.append(f"- commit_gate_details: `{payload['gates']['commit_gate_details']}`")
    if payload["gates"].get("improvement_vs_previous") is not None:
        lines.append(
            f"- improvement_vs_previous: `{100.0 * payload['gates']['improvement_vs_previous']:.2f}%`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare diffusion topology parity outputs")
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--cpp-root", required=True)
    parser.add_argument("--datasets", default="torus,sphere")
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", default="Diffusion topology parity")
    parser.add_argument("--previous-json", default="")
    parser.add_argument("--gate-improvement", type=float, default=0.20)
    parser.add_argument("--rcond", type=float, default=1e-10)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ref_root = Path(args.reference_root)
    cpp_root = Path(args.cpp_root)

    dataset_metrics: dict[str, Any] = {}
    warnings: list[str] = []
    for ds in datasets:
        metrics = compare_dataset(ref_root / ds, cpp_root / ds, args.rcond)
        dataset_metrics[ds] = metrics
        warnings.extend([f"{ds}: {w}" for w in metrics["warnings"]])

    summary = {
        "harmonic1_subspace_max_angle_deg": float(
            max(m["harmonic1_subspace_max_angle_deg"] for m in dataset_metrics.values())
        ),
        "harmonic1_procrustes_rel_error": float(
            max(m["harmonic1_procrustes_rel_error"] for m in dataset_metrics.values())
        ),
        "harmonic2_subspace_max_angle_deg": float(
            max(m["harmonic2_subspace_max_angle_deg"] for m in dataset_metrics.values())
        ),
        "harmonic2_procrustes_rel_error": float(
            max(m["harmonic2_procrustes_rel_error"] for m in dataset_metrics.values())
        ),
        "wedge_rel_error": float(max(m["wedge_rel_error"] for m in dataset_metrics.values())),
        "form2_spectrum_rel_error": float(
            max(m["form2_spectrum_rel_error"] for m in dataset_metrics.values())
        ),
        "circular_complex_correlation_min": float(
            min(m["circular_complex_correlation_min"] for m in dataset_metrics.values())
        ),
        "circular_wrapped_p95_rad_max": float(
            max(m["circular_wrapped_p95_rad_max"] for m in dataset_metrics.values())
        ),
    }
    summary["composite_score"] = composite_score(summary)

    final_pass = (
        summary["harmonic1_subspace_max_angle_deg"] <= 3.0
        and summary["harmonic1_procrustes_rel_error"] <= 0.05
        and summary["harmonic2_subspace_max_angle_deg"] <= 5.0
        and summary["harmonic2_procrustes_rel_error"] <= 0.10
        and summary["wedge_rel_error"] <= 0.10
        and summary["form2_spectrum_rel_error"] <= 0.10
        and summary["circular_complex_correlation_min"] >= 0.99
        and summary["circular_wrapped_p95_rad_max"] <= 0.10
    )

    previous_score = None
    if args.previous_json:
        prev_path = Path(args.previous_json)
        if prev_path.exists():
            previous = json.loads(prev_path.read_text())
            previous_score = float(previous.get("summary", {}).get("composite_score", math.nan))

    commit_pass = True
    improvement = None
    commit_details = "no previous baseline provided"
    if previous_score is not None and np.isfinite(previous_score):
        improvement = (previous_score - summary["composite_score"]) / max(previous_score, 1e-12)
        commit_pass = improvement >= args.gate_improvement
        commit_details = (
            f"required >= {100.0 * args.gate_improvement:.2f}% improvement; "
            f"observed {100.0 * improvement:.2f}%"
        )

    payload: dict[str, Any] = {
        "label": args.label,
        "reference_root": str(ref_root),
        "cpp_root": str(cpp_root),
        "datasets": dataset_metrics,
        "summary": summary,
        "warnings": warnings,
        "gates": {
            "final_pass": final_pass,
            "commit_gate_pass": commit_pass,
            "commit_gate_details": commit_details,
            "improvement_vs_previous": improvement,
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_markdown)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text(render_markdown(payload))

    print(f"composite_score={summary['composite_score']:.6f}")
    print(f"final_pass={final_pass}")
    print(f"commit_gate_pass={commit_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
