"""Evaluate best prompts from artifacts on test datasets.

For each config: resolve artifact dir → find best prompt → evaluate on test
datasets from config → update run_summary.json in artifact dir.

Usage:
    python eval_best.py --configs-dir configs/unblock
    python eval_best.py --configs-dir configs/unblock --test datasets/test_orig.json
    python eval_best.py --config configs/unblock/task-pro_reflect-max.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

_original_json_dump = json.dump
_original_json_dumps = json.dumps


def _patched_dump(obj, fp, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return _original_json_dump(obj, fp, **kwargs)


def _patched_dumps(obj, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return _original_json_dumps(obj, **kwargs)


json.dump = _patched_dump
json.dumps = _patched_dumps

from config import load_pipeline_config, PipelineConfig
from report import build_test_results, save_test_results_json, generate_heatmap_excel
from schema import load_and_validate_dataset


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate best prompts on test datasets using model from config.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Single config YAML.")
    group.add_argument("--configs-dir", type=Path, help="Directory with config YAMLs.")

    p.add_argument("--test", type=Path, nargs="*",
                   help="Override test dataset(s). If not set, uses test: from config.")
    p.add_argument("--rollout-mode", choices=["golden_path", "real_rollout"],
                   default="golden_path")
    return p.parse_args()


def find_best_prompt(artifact_dir: Path) -> Path | None:
    """Find best_system_prompt.txt in artifact dir."""
    candidate = artifact_dir / "best_system_prompt.txt"
    if candidate.exists():
        return candidate
    # Fallback: any best_*.txt
    found = sorted(artifact_dir.glob("best_*.txt"))
    return found[0] if found else None


def process_config(config_path: Path, test_override: list[Path] | None, rollout_mode: str) -> bool:
    """Load config, find artifact + best prompt, evaluate, update run_summary. Returns True on success."""
    cfg = load_pipeline_config(config_path)
    artifact_dir = cfg.effective_run_dir

    print(f"  run_name:     {cfg.run_name}")
    print(f"  artifact_dir: {artifact_dir}")

    if not artifact_dir.exists():
        print(f"  SKIP: artifact dir not found")
        return False

    prompt_path = find_best_prompt(artifact_dir)
    if not prompt_path:
        print(f"  SKIP: no best_*.txt in {artifact_dir}")
        return False

    test_paths = test_override or cfg.test
    if not test_paths:
        print(f"  SKIP: no test datasets in config and no --test flag")
        return False

    # Log full provider details.
    tp = cfg.task_provider
    print(f"  task_provider:")
    print(f"    name:        {tp.name}")
    print(f"    type:        {tp.type}")
    print(f"    model:       {tp.model}")
    print(f"    base_url:    {tp.base_url}")
    print(f"    timeout:     {tp.timeout_seconds}s")
    if tp.api_key_env:
        import os
        key_val = os.getenv(tp.api_key_env, "")
        key_preview = f"{key_val[:8]}...{key_val[-4:]}" if len(key_val) > 12 else ("SET" if key_val else "EMPTY")
        print(f"    api_key_env: {tp.api_key_env} ({key_preview})")
    if tp.auth_key_env:
        print(f"    auth_key_env: {tp.auth_key_env}")
    if tp.cert_file:
        print(f"    cert_file:   {tp.cert_file}")
    if tp.key_file:
        print(f"    key_file:    {tp.key_file}")
    if tp.extra_params:
        print(f"    extra_params: {tp.extra_params}")
    print(f"    retry:       max_attempts={tp.retry.max_attempts}, backoff={tp.retry.backoff_seconds}s")

    eval_temp = cfg.optimization.eval_temperature
    print(f"  eval_temperature: {eval_temp}")
    print(f"  rollout_mode:     {rollout_mode}")
    print(f"  prompt_file:      {prompt_path}")

    # Load test datasets.
    test_datasets: list[tuple[str, list]] = []
    for tpath in test_paths:
        ds = load_and_validate_dataset([tpath])
        test_datasets.append((tpath.stem, ds))
        print(f"  test_dataset:     {tpath} ({len(ds)} scenarios)")

    # Load prompt.
    candidate = {"system_prompt": prompt_path.read_text(encoding="utf-8")}
    print(f"  prompt_length:    {len(candidate['system_prompt'])} chars")
    print()

    # Create adapter based on adapter_type from config.
    if cfg.adapter_type == "kib_filter":
        from kib_adapter import KIBAdapter
        adapter = KIBAdapter(cfg.task_provider, temperature=eval_temp, top_p=0.1)
    elif cfg.adapter_type == "vis":
        from vis_adapter import VisAdapter
        adapter = VisAdapter(cfg.task_provider, temperature=eval_temp, top_p=0.1)
    else:
        from adapter import UnblockCardAdapter
        adapter = UnblockCardAdapter(cfg.task_provider, temperature=eval_temp, top_p=0.1)

    test_results: dict[str, dict[str, Any]] = {}
    try:
        for ds_name, ds_data in test_datasets:
            print(f"  [{ds_name}] ({len(ds_data)} scenarios)...", end=" ", flush=True)

            batch = adapter.evaluate(
                ds_data, candidate,
                capture_traces=True,
                rollout_mode=rollout_mode,
            )
            score = round(sum(batch.scores) / len(batch.scores), 4) if batch.scores else 0.0
            test_results[ds_name] = {"optimized": score, "size": len(ds_data)}
            print(f"score={score:.4f}")

            # Save heatmap + detailed results.
            step_results = build_test_results(batch, label="optimized")
            if step_results:
                save_test_results_json(
                    step_results,
                    artifact_dir / f"test_results_{ds_name}_optimized.json",
                )
                generate_heatmap_excel(
                    step_results,
                    output_path=artifact_dir / f"heatmap_{ds_name}_optimized.xlsx",
                    template_path=cfg.heatmap_template,
                )
    finally:
        adapter._adapter.close()

    # Update run_summary.json.
    summary_path = artifact_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "run_name": cfg.run_name,
            "task_lm": cfg.task_provider.model,
            "optimized_prompt": candidate["system_prompt"],
        }

    summary["test_results"] = test_results
    summary["test_eval_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary["test_rollout_mode"] = rollout_mode
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Updated: {summary_path}")
    return True


def main() -> None:
    args = parse_args()

    # Resolve test paths.
    test_override = None
    if args.test:
        test_override = [(ROOT / p).resolve() if not p.is_absolute() else p for p in args.test]

    # Collect configs.
    if args.config:
        configs = [args.config]
    else:
        configs = sorted(args.configs_dir.glob("*.yaml"))
        if not configs:
            raise SystemExit(f"No YAML configs found in {args.configs_dir}")

    print(f"Configs to process: {len(configs)}\n")

    ok = 0
    for config_path in configs:
        print(f"{'=' * 60}")
        print(f"Config: {config_path.name}")
        if process_config(config_path, test_override, args.rollout_mode):
            ok += 1
        print()

    print(f"Done: {ok}/{len(configs)} evaluated successfully.")


if __name__ == "__main__":
    main()
