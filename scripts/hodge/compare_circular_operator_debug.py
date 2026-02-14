#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_columns(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return {name: np.asarray(arr[name]) for name in arr.dtype.names}


def load_matrix(path: Path) -> np.ndarray:
    cols = load_columns(path)
    keys = [k for k in cols.keys() if k.startswith("c")]
    keys.sort(key=lambda x: int(x[1:]))
    return np.column_stack([cols[k] for k in keys]).astype(np.float64)


def load_evals(path: Path) -> np.ndarray:
    cols = load_columns(path)
    return cols["real"].astype(np.float64) + 1j * cols["imag"].astype(np.float64)


def load_positive_evals(path: Path) -> np.ndarray:
    cols = load_columns(path)
    evals = cols["real"].astype(np.float64) + 1j * cols["imag"].astype(np.float64)
    mask = cols["positive_imag"].astype(np.float64) > 0.5
    return evals[mask]


def rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    m = min(a.shape[1], b.shape[1])
    aa = a[:n, :m]
    bb = b[:n, :m]
    denom = max(np.linalg.norm(aa, ord="fro"), 1e-12)
    return float(np.linalg.norm(aa - bb, ord="fro") / denom)


def align_basis(
    reference_basis: np.ndarray, cpp_basis: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    n = min(reference_basis.shape[0], cpp_basis.shape[0])
    k_ref = reference_basis.shape[1]
    k_cpp = cpp_basis.shape[1]

    ref = reference_basis[:n, :k_ref]
    cpp = cpp_basis[:n, :k_cpp]

    transform, _, rank, singular_values = np.linalg.lstsq(cpp, ref, rcond=None)
    reconstructed = cpp @ transform
    denom = max(np.linalg.norm(ref, ord="fro"), 1e-12)
    recon_rel = float(np.linalg.norm(reconstructed - ref, ord="fro") / denom)

    cond = float("inf")
    if singular_values.size > 0:
        cond = float(singular_values[0] / max(singular_values[-1], 1e-12))

    stats = {
        "rows_used": float(n),
        "reference_cols": float(k_ref),
        "cpp_cols": float(k_cpp),
        "transform_rank": float(rank),
        "cpp_basis_condition_estimate": cond,
        "reconstruction_rel_fro": recon_rel,
    }
    return transform, stats


def transform_weak_to_reference_basis(cpp_weak: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return transform.conj().T @ cpp_weak @ transform


def eval_alignment_error(ref: np.ndarray, cpp: np.ndarray) -> dict[str, float]:
    k = min(ref.shape[0], cpp.shape[0])
    if k == 0:
        return {"mean_abs_diff": float("nan"), "p95_abs_diff": float("nan")}
    d = np.abs(ref[:k] - cpp[:k])
    return {
        "mean_abs_diff": float(np.mean(d)),
        "p95_abs_diff": float(np.percentile(d, 95.0)),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Circular Operator Debug Comparison")
    lines.append("")
    lines.append(f"- Reference dir: `{payload['reference_dir']}`")
    lines.append(f"- C++ dir: `{payload['cpp_dir']}`")
    lines.append("")
    lines.append("## Matrix Relative Frobenius Errors")
    lines.append("")
    for k, v in payload["matrix_rel_fro"].items():
        lines.append(f"- {k}: `{v:.6e}`")
    lines.append("")
    if payload.get("matrix_rel_fro_aligned"):
        lines.append("## Matrix Relative Frobenius Errors (Basis Aligned)")
        lines.append("")
        for k, v in payload["matrix_rel_fro_aligned"].items():
            lines.append(f"- {k}: `{v:.6e}`")
        lines.append("")
    if payload.get("basis_alignment"):
        lines.append("## Basis Alignment")
        lines.append("")
        stats = payload["basis_alignment"]
        lines.append(f"- rows used: `{int(stats['rows_used'])}`")
        lines.append(f"- reference columns: `{int(stats['reference_cols'])}`")
        lines.append(f"- cpp columns: `{int(stats['cpp_cols'])}`")
        lines.append(f"- transform rank: `{int(stats['transform_rank'])}`")
        lines.append(
            f"- cpp basis condition estimate: `{stats['cpp_basis_condition_estimate']:.6e}`"
        )
        lines.append(
            f"- reconstruction relative Frobenius error: `{stats['reconstruction_rel_fro']:.6e}`"
        )
        lines.append("")

    lines.append("## Positive-Imag Eigenvalue Alignment")
    lines.append("")
    for form, stats in payload["eval_alignment"].items():
        lines.append(
            f"- {form}: mean abs diff `{stats['mean_abs_diff']:.6e}`, "
            f"P95 abs diff `{stats['p95_abs_diff']:.6e}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare circular operator debug artifacts between reference and C++ outputs."
    )
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--cpp-dir", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    cpp_dir = Path(args.cpp_dir)

    matrix_pairs = {
        "function_gram": (
            ref_dir / "reference_function_gram.csv",
            cpp_dir / "function_gram.csv",
        ),
        "laplacian0_weak": (
            ref_dir / "reference_laplacian0_weak.csv",
            cpp_dir / "laplacian0_weak.csv",
        ),
        "form0_x_weak": (
            ref_dir / "reference_circular_operator_form0_x_weak.csv",
            cpp_dir / "circular_operator_form0_x_weak.csv",
        ),
        "form1_x_weak": (
            ref_dir / "reference_circular_operator_form1_x_weak.csv",
            cpp_dir / "circular_operator_form1_x_weak.csv",
        ),
        "form0_operator_weak": (
            ref_dir / "reference_circular_operator_form0_operator_weak.csv",
            cpp_dir / "circular_operator_form0_operator_weak.csv",
        ),
        "form1_operator_weak": (
            ref_dir / "reference_circular_operator_form1_operator_weak.csv",
            cpp_dir / "circular_operator_form1_operator_weak.csv",
        ),
    }

    matrix_rel_fro: dict[str, float] = {}
    for name, (rp, cp) in matrix_pairs.items():
        if not rp.exists() or not cp.exists():
            matrix_rel_fro[name] = float("nan")
            continue
        matrix_rel_fro[name] = rel_fro(load_matrix(rp), load_matrix(cp))

    matrix_rel_fro_aligned: dict[str, float] = {}
    basis_alignment: dict[str, float] = {}
    ref_basis_path = ref_dir / "reference_function_basis.csv"
    cpp_basis_path = cpp_dir / "function_basis.csv"
    if ref_basis_path.exists() and cpp_basis_path.exists():
        ref_basis = load_matrix(ref_basis_path)
        cpp_basis = load_matrix(cpp_basis_path)
        transform, basis_alignment = align_basis(ref_basis, cpp_basis)

        for name, (rp, cp) in matrix_pairs.items():
            if not rp.exists() or not cp.exists():
                matrix_rel_fro_aligned[name] = float("nan")
                continue
            ref_m = load_matrix(rp)
            cpp_m = load_matrix(cp)
            rdim = min(ref_m.shape[0], ref_m.shape[1], transform.shape[1])
            cdim = min(cpp_m.shape[0], cpp_m.shape[1], transform.shape[0])
            if rdim <= 0 or cdim <= 0:
                matrix_rel_fro_aligned[name] = float("nan")
                continue
            ref_crop = ref_m[:rdim, :rdim]
            cpp_crop = cpp_m[:cdim, :cdim]
            t_crop = transform[:cdim, :rdim]
            cpp_aligned = transform_weak_to_reference_basis(cpp_crop, t_crop)
            matrix_rel_fro_aligned[name] = rel_fro(ref_crop, cpp_aligned)

    eval_alignment: dict[str, dict[str, float]] = {}
    for form in [0, 1]:
        rp = ref_dir / f"reference_circular_operator_form{form}_evals.csv"
        cp = cpp_dir / f"circular_operator_form{form}_evals.csv"
        if not rp.exists() or not cp.exists():
            eval_alignment[f"form{form}"] = {
                "mean_abs_diff": float("nan"),
                "p95_abs_diff": float("nan"),
            }
            continue
        eval_alignment[f"form{form}"] = eval_alignment_error(
            load_positive_evals(rp), load_positive_evals(cp)
        )

    payload: dict[str, Any] = {
        "reference_dir": str(ref_dir.resolve()),
        "cpp_dir": str(cpp_dir.resolve()),
        "matrix_rel_fro": matrix_rel_fro,
        "matrix_rel_fro_aligned": matrix_rel_fro_aligned,
        "basis_alignment": basis_alignment,
        "eval_alignment": eval_alignment,
    }

    out_md = Path(args.output_markdown)
    out_json = Path(args.output_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload))
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"Wrote debug markdown: {out_md}")
    print(f"Wrote debug json: {out_json}")


if __name__ == "__main__":
    main()
