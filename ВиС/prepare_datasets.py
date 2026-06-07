"""Merge v10 + v2 datasets, stratified split into train/val.

Usage:
    python ВиС/prepare_datasets.py

Outputs:
    ВиС/datasets/train.json  (~80%)
    ВиС/datasets/val.json    (~20%)

Stratification key: base_scenario × action_types present in dialog.
Original files (v10, v2) are kept as-is for final testing.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

SEED = 42
VAL_RATIO = 0.2

ROOT = Path(__file__).parent
DATSETS_DIR = ROOT / "Датасеты"
OUT_DIR = ROOT / "datasets"


def base_scenario(scenario_id: str) -> str:
    """scenario_1/modification_3 -> scenario_1"""
    return re.split(r"/", scenario_id)[0]


def get_action_types(item: dict) -> str:
    """Sorted comma-joined set of agent action types in the dialog."""
    types = sorted({
        step["type"]
        for step in item["steps"]
        if step.get("role") == "agent" and "type" in step
    })
    return ",".join(types) if types else "_none_"


def stratification_key(item: dict) -> str:
    return f"{base_scenario(item['scenario_id'])}|{get_action_types(item)}"


def print_split_summary(train: list[dict], val: list[dict]) -> None:
    for name, data in [("train", train), ("val", val)]:
        total = len(data)
        scenarios = Counter(base_scenario(d["scenario_id"]) for d in data)
        types: Counter[str] = Counter()
        for d in data:
            for step in d["steps"]:
                if step.get("role") == "agent" and "type" in step:
                    types[step["type"]] += 1

        print(f"\n{'='*50}")
        print(f"  {name}: {total} items")
        print(f"{'='*50}")
        print("  Scenarios:")
        for s, c in sorted(scenarios.items()):
            print(f"    {s:<15s} {c:>4d}  ({c/total*100:5.1f}%)")
        print("  Action types:")
        for t, c in types.most_common():
            print(f"    {t:<20s} {c:>4d}  ({c/sum(types.values())*100:5.1f}%)")


def main() -> None:
    v10 = json.loads((DATSETS_DIR / "dataset_ViS_v10.json").read_text(encoding="utf-8"))
    v2 = json.loads((DATSETS_DIR / "test_dataset_modification_v2.json").read_text(encoding="utf-8"))

    # Merge all — even overlapping dialog_ids have different content.
    all_data = v10 + v2
    print(f"Total items: {len(all_data)} (v10={len(v10)}, v2={len(v2)})")

    # Group by stratification key (base_scenario × action_types).
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in all_data:
        groups[stratification_key(item)].append(item)

    rng = Random(SEED)
    train: list[dict] = []
    val: list[dict] = []

    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * VAL_RATIO))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    # Write.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val)]:
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}: {len(data)} items -> {path}")

    print_split_summary(train, val)

    # Validate with pipeline schema.
    print("\nValidation:")
    sys.path.insert(0, str(ROOT.parent))
    from schema import load_and_validate_dataset

    for name in ("train", "val"):
        path = OUT_DIR / f"{name}.json"
        loaded = load_and_validate_dataset([path])
        print(f"  {name}: {len(loaded)} validated OK")

    print(f"\nStratification groups: {len(groups)}")
    print("Original v10 and v2 files kept for final testing.")


if __name__ == "__main__":
    main()
