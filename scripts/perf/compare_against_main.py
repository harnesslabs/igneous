#!/usr/bin/env python3
"""Compare benchmark results and emit markdown + JSON summaries.

Supports Google Benchmark JSON/TXT and metrics.csv inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

AGGREGATE_SUFFIX_RE = re.compile(r"_(mean|median|stddev|cv)$")
GBENCH_TEXT_RE = re.compile(r"^(bench\S+)\s+([0-9.+\-eE]+)\s+(ns|us|ms|s)\b")
GEOMETRY_ROW_RE = re.compile(
    r"^Grid\s+(\d+x\d+)\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)


def to_ns(value: float, unit: str) -> float:
    factors = {
        "ns": 1.0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }
    return value * factors.get(unit, 1.0)


def normalize_benchmark_id(raw: str) -> str:
    bench_id = raw.strip()
    if not bench_id:
        return bench_id

    bench_id = AGGREGATE_SUFFIX_RE.sub("", bench_id)

    if "_bench_" in bench_id:
        tail = bench_id.rsplit("_bench_", 1)[1]
        bench_id = tail if tail.startswith("bench_") else f"bench_{tail}"
    elif bench_id.count("bench_") > 1:
        bench_id = bench_id[bench_id.rfind("bench_") :]

    bench_id = bench_id.replace("__", "_")
    return bench_id


def human_ns(ns_value: float | None) -> str:
    if ns_value is None or math.isnan(ns_value):
        return "n/a"
    if ns_value >= 1e9:
        return f"{ns_value / 1e9:.3f} s"
    if ns_value >= 1e6:
        return f"{ns_value / 1e6:.3f} ms"
    if ns_value >= 1e3:
        return f"{ns_value / 1e3:.3f} us"
    return f"{ns_value:.3f} ns"


def parse_google_benchmark_json(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text())
    means: Dict[str, List[float]] = {}
    raws: Dict[str, List[float]] = {}

    for row in payload.get("benchmarks", []):
        name = row.get("run_name") or row.get("name")
        if not name:
            continue

        run_type = row.get("run_type")
        agg_name = row.get("aggregate_name")
        value = row.get("real_time")
        if value is None:
            continue

        bench_id = normalize_benchmark_id(str(name))
        ns_value = to_ns(float(value), str(row.get("time_unit", "ns")))

        if run_type == "aggregate":
            if agg_name == "mean":
                means.setdefault(bench_id, []).append(ns_value)
        else:
            raws.setdefault(bench_id, []).append(ns_value)

    merged: Dict[str, float] = {}
    for bench_id in sorted(set(means) | set(raws)):
        values = means.get(bench_id) or raws.get(bench_id)
        if values:
            merged[bench_id] = float(mean(values))
    return merged


def parse_benchmark_text(path: Path) -> Dict[str, float]:
    rows: Dict[str, List[float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        m = GBENCH_TEXT_RE.match(line)
        if m:
            name = normalize_benchmark_id(m.group(1))
            if name.endswith("_median") or name.endswith("_stddev") or name.endswith("_cv"):
                continue
            ns_value = to_ns(float(m.group(2)), m.group(3))
            rows.setdefault(name, []).append(ns_value)
            continue

        g = GEOMETRY_ROW_RE.match(line)
        if g:
            grid = g.group(1)
            topo_ms = float(g.group(2))
            curv_ms = float(g.group(3))
            flow_ms = float(g.group(4))
            frame_ms = topo_ms + curv_ms + flow_ms

            rows.setdefault(f"bench_geometry_structure_ms/{grid}", []).append(to_ns(topo_ms, "ms"))
            rows.setdefault(f"bench_geometry_curvature_ms/{grid}", []).append(to_ns(curv_ms, "ms"))
            rows.setdefault(f"bench_geometry_flow_ms/{grid}", []).append(to_ns(flow_ms, "ms"))
            rows.setdefault(f"bench_geometry_frame_ms/{grid}", []).append(to_ns(frame_ms, "ms"))

    return {bench_id: float(mean(values)) for bench_id, values in rows.items() if values}


def parse_metrics_csv(path: Path, baseline_commit: str | None) -> Dict[str, float]:
    rows: Dict[str, List[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if baseline_commit and row.get("commit") != baseline_commit:
                continue

            mean_ns_raw = row.get("mean_ns")
            if not mean_ns_raw:
                continue

            try:
                mean_ns_value = float(mean_ns_raw)
            except ValueError:
                continue

            benchmark_id = normalize_benchmark_id(row.get("benchmark_id", ""))
            if not benchmark_id:
                continue

            dataset_id = (row.get("dataset_id") or "").strip()
            if dataset_id and dataset_id not in {"", "synthetic"}:
                benchmark_id = f"{benchmark_id}/{dataset_id}"

            rows.setdefault(benchmark_id, []).append(mean_ns_value)

    return {bench_id: float(mean(values)) for bench_id, values in rows.items() if values}


def load_benchmarks(path: Path, baseline_commit: str | None) -> Tuple[Dict[str, float], str | None]:
    if not path.exists():
        return {}, f"missing input: {path}"

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return parse_google_benchmark_json(path), None
        if suffix == ".txt":
            return parse_benchmark_text(path), None
        if suffix == ".csv":
            return parse_metrics_csv(path, baseline_commit), None
    except Exception as exc:  # pragma: no cover - defensive path
        return {}, f"failed to parse {path}: {exc}"

    return {}, f"unsupported input format: {path}"


def compare(baseline: Dict[str, float], current: Dict[str, float]) -> List[dict]:
    rows = []
    for bench_id in sorted(set(baseline) | set(current)):
        base_ns = baseline.get(bench_id)
        curr_ns = current.get(bench_id)
        if base_ns is not None and curr_ns is not None and base_ns != 0:
            delta_pct = ((curr_ns - base_ns) / base_ns) * 100.0
            comparable = True
        else:
            delta_pct = None
            comparable = False

        if not comparable:
            status = "not comparable"
        elif delta_pct is not None and delta_pct < 0:
            status = "improved"
        elif delta_pct is not None and delta_pct > 0:
            status = "regressed"
        else:
            status = "flat"

        rows.append(
            {
                "benchmark_id": bench_id,
                "baseline_ns": base_ns,
                "current_ns": curr_ns,
                "delta_pct": delta_pct,
                "comparable": comparable,
                "status": status,
            }
        )
    return rows


def render_markdown(
    label: str,
    baseline_path: Path,
    current_path: Path,
    baseline_commit: str,
    rows: List[dict],
    warnings: List[str],
) -> str:
    comparable = [r for r in rows if r["comparable"]]
    improved = [r for r in comparable if r["delta_pct"] is not None and r["delta_pct"] < 0]
    regressed = [r for r in comparable if r["delta_pct"] is not None and r["delta_pct"] > 0]

    top_wins = sorted(improved, key=lambda r: r["delta_pct"])[:5]
    top_regs = sorted(regressed, key=lambda r: r["delta_pct"], reverse=True)[:5]

    lines: List[str] = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"- Baseline source: `{baseline_path}`")
    lines.append(f"- Current source: `{current_path}`")
    lines.append(f"- Baseline commit: `{baseline_commit}`")
    lines.append(f"- Comparable benchmarks: `{len(comparable)}/{len(rows)}`")
    lines.append(f"- Improved: `{len(improved)}` | Regressed: `{len(regressed)}`")

    if warnings:
        lines.append("- Notes:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    lines.append("")

    if top_wins:
        lines.append("### Top Wins")
        lines.append("")
        lines.append("| Benchmark | Baseline | Current | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in top_wins:
            lines.append(
                f"| `{row['benchmark_id']}` | {human_ns(row['baseline_ns'])} | {human_ns(row['current_ns'])} | {row['delta_pct']:.2f}% |"
            )
        lines.append("")

    if top_regs:
        lines.append("### Top Regressions")
        lines.append("")
        lines.append("| Benchmark | Baseline | Current | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in top_regs:
            lines.append(
                f"| `{row['benchmark_id']}` | {human_ns(row['baseline_ns'])} | {human_ns(row['current_ns'])} | +{row['delta_pct']:.2f}% |"
            )
        lines.append("")

    lines.append("### Full Comparison")
    lines.append("")
    lines.append("| Benchmark | Baseline | Current | Delta | Status |")
    lines.append("| --- | ---: | ---: | ---: | --- |")

    for row in rows:
        delta = "n/a"
        if row["delta_pct"] is not None:
            delta = f"{row['delta_pct']:+.2f}%"
        lines.append(
            f"| `{row['benchmark_id']}` | {human_ns(row['baseline_ns'])} | {human_ns(row['current_ns'])} | {delta} | {row['status']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark outputs against a baseline")
    parser.add_argument("--baseline", required=True, help="Baseline input file (json/txt/csv)")
    parser.add_argument("--current", required=True, help="Current input file (json/txt/csv)")
    parser.add_argument("--label", default="Benchmark comparison")
    parser.add_argument("--baseline-commit", default="e761562")
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    baseline, baseline_warning = load_benchmarks(baseline_path, args.baseline_commit)
    current, current_warning = load_benchmarks(current_path, None)

    warnings: List[str] = []
    if baseline_warning:
        warnings.append(baseline_warning)
    if current_warning:
        warnings.append(current_warning)

    rows = compare(baseline, current)

    markdown = render_markdown(
        label=args.label,
        baseline_path=baseline_path,
        current_path=current_path,
        baseline_commit=args.baseline_commit,
        rows=rows,
        warnings=warnings,
    )

    output_markdown = Path(args.output_markdown)
    output_json = Path(args.output_json)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown)

    payload = {
        "label": args.label,
        "baseline_source": str(baseline_path),
        "current_source": str(current_path),
        "baseline_commit": args.baseline_commit,
        "warnings": warnings,
        "summary": {
            "total": len(rows),
            "comparable": sum(1 for r in rows if r["comparable"]),
            "improved": sum(1 for r in rows if r["status"] == "improved"),
            "regressed": sum(1 for r in rows if r["status"] == "regressed"),
            "flat": sum(1 for r in rows if r["status"] == "flat"),
            "not_comparable": sum(1 for r in rows if r["status"] == "not comparable"),
        },
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
