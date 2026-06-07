"""Convert КИБ dict-indexed dataset to GEPA pipeline format with stratified split.

Input:  LLM_Фильтр_Тестовый_Датасет_ИФТ_v4.json (222 scenarios, dict-indexed)
Output: КИБ/datasets/{train,val,test}.json — pipeline-format with stratified 70/15/15 split.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

ROOT = Path(__file__).parent
SRC = ROOT / "Датасеты" / "LLM_Фильтр_Тестовый_Датасет_ИФТ_v4.json"
OUT_DIR = ROOT / "datasets"


def convert_to_pipeline_format(raw: dict) -> list[dict]:
    """Convert dict-indexed КИБ format to pipeline scenario list."""
    types = raw["type"]
    messages = raw["messages"]
    n = len(types)

    scenarios: list[dict] = []
    for i in range(n):
        idx = str(i)
        label = types[idx]
        user_messages = messages[idx]

        # Build pipeline scenario: user message → agent label response
        steps = []
        for j, msg in enumerate(user_messages):
            steps.append({
                "step": j * 2 + 1,
                "role": msg["role"],
                "message": msg["content"],
            })

        # Agent step: expected label with exact-match regex
        steps.append({
            "step": len(user_messages) * 2,
            "role": "agent",
            "message": label,
            "additional_check": f"^{label}$",
        })

        scenarios.append({
            "scenario_id": f"kib_{i}",
            "dialog_id": f"kib_{i}",
            "steps": steps,
        })

    return scenarios


def stratified_split(
    scenarios: list[dict],
    labels: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split scenarios into train/val/test with stratification by label."""
    rng = random.Random(seed)

    # Group indices by label
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_label[lbl].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for lbl, indices in sorted(by_label.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    # Shuffle each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        [scenarios[i] for i in train_idx],
        [scenarios[i] for i in val_idx],
        [scenarios[i] for i in test_idx],
    )


def main() -> None:
    raw: dict = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"Loaded {len(raw['type'])} entries from {SRC.name}")

    scenarios = convert_to_pipeline_format(raw)
    labels = [raw["type"][str(i)] for i in range(len(raw["type"]))]

    train, val, test = stratified_split(scenarios, labels, TRAIN_RATIO, VAL_RATIO, SEED)

    # Print distribution
    from collections import Counter
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(s["steps"][-1]["message"] for s in split)
        print(f"  {name}: {len(split)} — {dict(sorted(dist.items()))}")

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {path}")

    # Validate
    sys.path.insert(0, str(ROOT.parent))
    from schema import load_and_validate_dataset

    for name in ("train", "val", "test"):
        path = OUT_DIR / f"{name}.json"
        loaded = load_and_validate_dataset([path])
        print(f"  ✓ {name}: {len(loaded)} validated")


if __name__ == "__main__":
    main()
