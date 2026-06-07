"""Entry point for prompt optimization via gepa.optimize().

Usage:
    python run.py --config configs/unblock/pro_gepa_default.yaml

All settings (including cert/key paths) live in the YAML config.
Use --max-calls / --patience to override budget params without editing YAML.
"""

from __future__ import annotations
import random
from typing import Any
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from datetime import datetime
from pathlib import Path

# gepa writes JSON artifacts with the default ensure_ascii=True, which escapes
# Cyrillic into \uXXXX sequences.  We patch json.dump/dumps to default to
# ensure_ascii=False instead.  Using setdefault means any caller that explicitly
# passes ensure_ascii=True still gets the original behaviour — only the default
# changes.  This affects the entire process; a safer alternative would require
# gepa to expose a JSON encoder parameter.
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

import openpyxl

from gepa import optimize, NoImprovementStopper, MaxMetricCallsStopper
from adapter import UnblockCardAdapter
from config import load_pipeline_config, PipelineConfig
from llm import GigaChatLanguageModel, TokenUsage
from report import build_test_results, save_test_results_json, generate_heatmap_excel, compute_classification_metrics
from schema import load_and_validate_dataset


_EXCEL_HEADERS = [
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
    "test_results",
    "base_prompt",
    "optimized_prompt",
]


def _build_candidate_selector(cfg: PipelineConfig):
    """Build candidate selector from config, allowing custom top_k."""
    strategy = cfg.optimization.candidate_selection_strategy
    if strategy == "top_k_pareto":
        from gepa.strategies.candidate_selector import TopKParetoCandidateSelector
        return TopKParetoCandidateSelector(
            k=cfg.optimization.top_k,
            rng=random.Random(cfg.optimization.seed),
        )
    # For other strategies, pass the string — gepa resolves it internally.
    return strategy


def log_to_excel(path: Path, row: dict) -> None:
    """Append one row to the experiments Excel file, creating it if needed."""
    if path.exists():
        wb = openpyxl.load_workbook(path)
        ws = wb.active
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "experiments"
        ws.append(_EXCEL_HEADERS)

    ws.append([row.get(h) for h in _EXCEL_HEADERS])
    wb.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the card-unblocking system prompt via gepa.optimize().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--max-calls", type=int, default=None, help="Override optimization.max_calls.")
    parser.add_argument("--patience", type=int, default=None, help="Override optimization.patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_pipeline_config(args.config)

    if args.max_calls is not None:
        cfg.optimization.max_calls = args.max_calls
    if args.patience is not None:
        cfg.optimization.patience = args.patience

    trainset: list = load_and_validate_dataset(cfg.train)
    valset: list | None = load_and_validate_dataset(cfg.val) if cfg.val else None
    # Load each test dataset separately for per-dataset metrics.
    test_datasets: list[tuple[str, list]] = []
    for test_path in cfg.test:
        ds = load_and_validate_dataset([test_path])
        test_datasets.append((test_path.stem, ds))
    if cfg.seed_prompts:
        seed_candidate = {
            name: path.read_text(encoding="utf-8")
            for name, path in cfg.seed_prompts.items()
        }
    else:
        seed_candidate = {"system_prompt": cfg.seed_prompt.read_text(encoding="utf-8")}

    if cfg.adapter_type == "kib_filter":
        from kib_adapter import KIBAdapter
        adapter = KIBAdapter(cfg.task_provider, temperature=cfg.optimization.eval_temperature, top_p=0.1)
    elif cfg.adapter_type == "vis":
        from vis_adapter import VisAdapter
        adapter = VisAdapter(cfg.task_provider, temperature=cfg.optimization.eval_temperature, top_p=0.1)
    else:
        adapter = UnblockCardAdapter(cfg.task_provider, temperature=cfg.optimization.eval_temperature, top_p=0.1)

    reflection_lm = GigaChatLanguageModel(
        cfg.effective_reflection_provider,
        temperature=cfg.mutator.temperature,
    )
    reflection_usage = TokenUsage()

    run_dir = str(cfg.effective_run_dir)
    token_usage_path = Path(run_dir) / "token_usage.json"
    # Restore token counters from a previous (interrupted) run.
    if token_usage_path.exists():
        saved = json.loads(token_usage_path.read_text(encoding="utf-8"))
        adapter.input_tokens = saved.get("eval_input_tokens", 0)
        adapter.output_tokens = saved.get("eval_output_tokens", 0)
        reflection_lm.input_tokens = saved.get("reflection_input_tokens", 0)
        reflection_lm.output_tokens = saved.get("reflection_output_tokens", 0)
        print(f"Restored token usage from previous run: {sum(saved.values()):,} total")

    try:
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            reflection_prompt_template=cfg.mutator.reflection_prompt_template,
            reflection_minibatch_size=cfg.optimization.minibatch_size,
            module_selector=cfg.module_selector,
            candidate_selection_strategy=_build_candidate_selector(cfg),
            use_merge=cfg.optimization.use_merge,
            stop_callbacks=[
                NoImprovementStopper(cfg.optimization.patience),
                MaxMetricCallsStopper(cfg.optimization.max_calls),
            ],
            run_dir=run_dir,
            seed=cfg.optimization.seed,
            display_progress_bar=True,
            cache_evaluation=True,
        )

        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, text in result.best_candidate.items():
            (out_dir / f"best_{name}.txt").write_text(text, encoding="utf-8")
        best_prompt = "\n\n".join(result.best_candidate.values())

        best_val_score = result.val_aggregate_scores[result.best_idx]
        base_score = result.val_aggregate_scores[0]
        accepted_mutations = result.num_candidates - 1

        # Epoch = one full pass through the trainset.
        # total_metric_calls counts all evaluations (train minibatch + val).
        # Subtract full valset evaluations to isolate train-side evals.
        total_calls = result.total_metric_calls or 0
        full_val_evals = result.num_full_val_evals or 0
        valset_size = len(valset) if valset else 0
        train_evals = total_calls - full_val_evals * valset_size
        trainset_size = len(trainset)
        epochs = round(train_evals / trainset_size, 2) if trainset_size > 0 else 0

        # Final evaluation on each test dataset separately.
        test_results: dict[str, dict[str, float]] = {}
        if test_datasets:
            best_candidate = result.best_candidate
            if isinstance(best_candidate, str):
                best_candidate = {"system_prompt": best_candidate}

            for ds_name, ds_data in test_datasets:
                print(f"\nEvaluating on test set '{ds_name}' ({len(ds_data)} scenarios)...")
                base_batch = adapter.evaluate(ds_data, seed_candidate, capture_traces=True, rollout_mode="golden_path")
                base_score_test = round(sum(base_batch.scores) / len(base_batch.scores), 4)

                best_batch = adapter.evaluate(ds_data, best_candidate, capture_traces=True, rollout_mode="golden_path")
                optimized_score_test = round(sum(best_batch.scores) / len(best_batch.scores), 4)

                ds_result: dict[str, Any] = {
                    "base": base_score_test,
                    "optimized": optimized_score_test,
                    "size": len(ds_data),
                }

                # Classification metrics (КИБ adapter)
                base_cls = compute_classification_metrics(base_batch)
                opt_cls = compute_classification_metrics(best_batch)
                if base_cls:
                    ds_result["base_metrics"] = base_cls
                if opt_cls:
                    ds_result["optimized_metrics"] = opt_cls

                test_results[ds_name] = ds_result
                print(f"  {ds_name}: base={base_score_test:.4f}, optimized={optimized_score_test:.4f}")
                if opt_cls:
                    print(f"    optimized macro: P={opt_cls['macro_precision']:.4f} R={opt_cls['macro_recall']:.4f} F1={opt_cls['macro_f1']:.4f}")
                    print(f"    optimized micro: P={opt_cls['micro_precision']:.4f} R={opt_cls['micro_recall']:.4f} F1={opt_cls['micro_f1']:.4f}")

                # Save per-step results and heatmaps for each dataset.
                for label, batch in [("base", base_batch), ("optimized", best_batch)]:
                    step_results = build_test_results(batch, label=label)
                    if step_results:
                        save_test_results_json(step_results, out_dir / f"test_results_{ds_name}_{label}.json")
                        generate_heatmap_excel(
                            step_results,
                            output_path=out_dir / f"heatmap_{ds_name}_{label}.xlsx",
                            template_path=cfg.heatmap_template,
                        )

        # Collect reflection token usage.
        reflection_usage = TokenUsage(reflection_lm.input_tokens, reflection_lm.output_tokens)
        total_input = adapter.input_tokens + reflection_usage.input_tokens
        total_output = adapter.output_tokens + reflection_usage.output_tokens

        # Format test_results for Excel (compact JSON string).
        test_results_str = json.dumps(test_results, ensure_ascii=False) if test_results else None

        excel_path = cfg.run_dir / "experiments.xlsx"
        summary_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "run_name": cfg.run_name,
            "task_lm": cfg.task_provider.model,
            "reflection_lm": cfg.effective_reflection_provider.model,
            "mutator_type": cfg.mutator.type,
            "accepted_mutations": accepted_mutations,
            "epochs": epochs,
            "eval_input_tokens": adapter.input_tokens,
            "eval_output_tokens": adapter.output_tokens,
            "reflection_input_tokens": reflection_usage.input_tokens,
            "reflection_output_tokens": reflection_usage.output_tokens,
            "total_tokens": total_input + total_output,
            "test_results": test_results_str,
            "base_prompt": "\n\n".join(seed_candidate.values()),
            "optimized_prompt": best_prompt,
        }
        log_to_excel(excel_path, summary_row)

        # run_summary.json stores test_results as a proper dict (not string).
        summary_json = {**summary_row, "test_results": test_results}
        summary_path = out_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\nRun:                     {cfg.run_name}")
        print(f"Base prompt metric:      {base_score:.3f}")
        print(f"Optimized prompt metric: {best_val_score:.3f}")
        for ds_name, scores in test_results.items():
            print(f"Test '{ds_name}' ({scores['size']}):  base={scores['base']:.4f}  optimized={scores['optimized']:.4f}")
        print(f"Accepted mutations:      {accepted_mutations}")
        print(f"Epochs:                  {epochs}")
        print(f"Eval tokens:             {adapter.input_tokens + adapter.output_tokens:,} (in: {adapter.input_tokens:,} / out: {adapter.output_tokens:,})")
        print(f"Reflection tokens:       {reflection_usage.total_tokens:,} (in: {reflection_usage.input_tokens:,} / out: {reflection_usage.output_tokens:,})")
        print(f"Total tokens:            {total_input + total_output:,}")
        print(f"Results saved to:        {out_dir}")
        print(f"Run summary:             {summary_path}")
        print(f"Excel log:               {excel_path}")
    finally:
        # Persist token counters so they survive restarts.
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        token_usage_path.write_text(json.dumps({
            "eval_input_tokens": adapter.input_tokens,
            "eval_output_tokens": adapter.output_tokens,
            "reflection_input_tokens": reflection_lm.input_tokens,
            "reflection_output_tokens": reflection_lm.output_tokens,
        }, indent=2), encoding="utf-8")

        adapter._adapter.close()
        reflection_lm._adapter.close()


if __name__ == "__main__":
    main()
