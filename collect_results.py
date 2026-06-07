"""Collect run_summary.json from all artifact subdirectories into a single experiments.xlsx.

Usage:
    python collect_results.py                          # default: artifacts/gepa_pipeline
    python collect_results.py --artifacts-dir path/to/artifacts
    python collect_results.py --output results.xlsx    # custom output path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl

# Fixed columns that always appear.
_FIXED_HEADERS = [
    "timestamp",
    "run_name",
    "task_lm",
    "reflection_lm",
    "mutator_type",
    "accepted_mutations",
    "epochs",
    "eval_input_tokens",
    "eval_output_tokens",
    "reflection_input_tokens",
    "reflection_output_tokens",
    "total_tokens",
]

_TAIL_HEADERS = [
    "base_prompt",
    "optimized_prompt",
]


def _flatten_test_results(data: dict) -> dict:
    """Flatten test_results dict into per-dataset base/optimized columns."""
    flat: dict[str, object] = {}
    test_results = data.get("test_results")
    if isinstance(test_results, str):
        try:
            test_results = json.loads(test_results)
        except (json.JSONDecodeError, TypeError):
            test_results = None
    if isinstance(test_results, dict):
        for ds_name, scores in test_results.items():
            if isinstance(scores, dict):
                flat[f"base_{ds_name}"] = scores.get("base")
                flat[f"optimized_{ds_name}"] = scores.get("optimized")
                flat[f"size_{ds_name}"] = scores.get("size")
    return flat


def collect(artifacts_dir: Path, output_path: Path) -> int:
    summaries: list[dict] = []

    for summary_file in sorted(artifacts_dir.rglob("run_summary.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            summaries.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {summary_file}: {e}")

    if not summaries:
        print(f"No run_summary.json found in {artifacts_dir}")
        return 0

    # Deduplicate by (run_name, timestamp).
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for s in summaries:
        key = (s.get("run_name", ""), s.get("timestamp", ""))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    # Flatten test_results and discover all metric columns.
    rows: list[dict] = []
    metric_columns: list[str] = []
    for data in unique:
        flat = _flatten_test_results(data)
        for col in flat:
            if col not in metric_columns:
                metric_columns.append(col)
        rows.append({**data, **flat})

    headers = _FIXED_HEADERS + sorted(metric_columns) + _TAIL_HEADERS

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "experiments"
    ws.append(headers)

    for row in rows:
        ws.append([row.get(h) for h in headers])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"Collected {len(rows)} runs → {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run summaries into experiments.xlsx")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/gepa_pipeline"),
        help="Root directory to search for run_summary.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Excel path (default: <artifacts-dir>/experiments.xlsx)",
    )
    args = parser.parse_args()

    output = args.output or args.artifacts_dir / "experiments.xlsx"
    collect(args.artifacts_dir, output)


if __name__ == "__main__":
    main()
