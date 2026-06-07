# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run optimization pipeline
python run.py --config configs/unblock/pro_gepa_default.yaml
python run.py --config configs/vis/vis_max_max.yaml --max-calls 1000 --patience 10

# Run tests
pytest tests/test_fixes.py -v

# Collect all run summaries into experiments.xlsx
python collect_results.py --artifacts-dir artifacts/gepa_pipeline --output results.xlsx

# Install dependencies
pip install -r requirements.txt
```

## Architecture

This is a **prompt optimization pipeline** for Russian chatbots using the [gepa](https://github.com/gepa-ai/gepa) library. It evaluates system prompts against dialog datasets, mutates them via LLM reflection, and selects the best candidates using Pareto-based strategies.

### Data Flow

```
YAML Config → PipelineConfig (Pydantic) → load datasets
    → gepa.optimize(seed, trainset, valset, adapter.evaluate, reflection_lm)
    → loop: evaluate → collect failures → reflect → mutate prompt → re-evaluate
    → best candidate + test evaluation → artifacts
```

### Core Modules

- **`run.py`** — Entry point. Orchestrates `gepa.optimize()`, handles token tracking, artifact generation, multi-dataset test evaluation.
- **`llm.py`** — LLM abstraction. `ModelProviderConfig` for GigaChat (mTLS) and OpenAI-compatible providers. `GigaChatLanguageModel` wraps adapters as gepa's `reflection_lm`. Exponential backoff retry (5s × 2^attempt, 6 attempts).
- **`config.py`** — Pydantic models: `PipelineConfig`, `MutatorConfig`, `OptimizationConfig`. YAML loading with path resolution relative to pipeline root.
- **`schema.py`** — Dataset validation: `ScenarioData` (inline steps or dialogs list), `DialogData`, `StepData` (with optional `additional_check` regex).
- **`report.py`** — Builds per-step results, generates scenario×action_type heatmap Excel files with openpyxl.

### Task Adapters (implement gepa's GEPAAdapter protocol)

| Adapter | File | Task | Mode |
|---|---|---|---|
| `UnblockCardAdapter` | `adapter.py` | Multi-turn dialog with regex validation | Single-component |
| `KIBAdapter` | `kib_adapter.py` | Single-turn classification (structured output) | Single-component |
| `VisAdapter` | `vis_adapter.py` | Document routing | Multi-component |

### Multi-Component Optimization (ВиС)

- Config uses `seed_prompts: {component_name: path}` instead of `seed_prompt`
- `module_selector: "round_robin"` mutates one component per iteration
- `make_reflective_dataset()` injects other components' text as context with boundary rules to prevent duplication between routing_logic and doc_catalog

### Evaluation Modes

- **golden_path** (default) — Each step evaluated independently using expected messages from dataset
- **real_rollout** — Cascading: LLM responses from earlier steps feed into later steps

### Key Design Details

- `json.dump`/`dumps` patched globally to default `ensure_ascii=False` (preserves Cyrillic)
- Thread-safe evaluation with `ThreadPoolExecutor(max_workers=5)`
- Token usage persisted to `token_usage.json` across process restarts
- Reflection templates must contain `<curr_param>` and `<side_info>` placeholders
- `adapter_type` values: `"unblock_card"`, `"kib_filter"`, `"vis"`

## Output Artifacts

Generated in `{run_dir}/{run_name}/`: `best_*.txt` (optimized prompts), `events.jsonl`, `candidate_tree.html`, `test_results_*.json`, `heatmap_*.xlsx`, `run_summary.json`, `token_usage.json`.

Metric: accuracy = successful_steps / total_steps (0.0–1.0).
