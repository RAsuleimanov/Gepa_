"""Pydantic config model for the gepa pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from llm import ModelProviderConfig


class MutatorConfig(BaseModel):
    type: Literal["gepa_default"] = "gepa_default"
    """gepa's built-in InstructionProposalSignature."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    reflection_prompt_template: str | None = None
    """Only for gepa_default. Must contain <curr_param> and <side_info>."""

    @model_validator(mode="after")
    def check_template_placeholders(self) -> "MutatorConfig":
        if self.type == "gepa_default" and self.reflection_prompt_template is not None:
            missing = [p for p in ("<curr_param>", "<side_info>")
                       if p not in self.reflection_prompt_template]
            if missing:
                raise ValueError(f"reflection_prompt_template must contain {', '.join(missing)}")
        return self


class OptimizationConfig(BaseModel):
    max_calls: int = Field(default=500, ge=1)
    patience: int = Field(default=5, ge=1)
    minibatch_size: int | None = Field(default=None, ge=1)
    """
    Scenarios per iteration for reflection/mutation.
    None = full trainset each iteration (expensive with large datasets).
    Recommended: 10-20% of trainset size.
    """
    candidate_selection_strategy: Literal[
        "pareto", "current_best", "epsilon_greedy", "top_k_pareto"
    ] = "pareto"
    top_k: int = Field(default=5, ge=1)
    """k parameter for top_k_pareto strategy. Ignored for other strategies."""
    use_merge: bool = True
    eval_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 0


class PipelineConfig(BaseModel):
    run_name: str = Field(min_length=1)

    adapter_type: Literal["unblock_card", "kib_filter", "vis"] = "unblock_card"
    """Which task adapter to use: unblock_card (multi-turn dialog) or kib_filter (single-turn classification)."""

    task_provider: ModelProviderConfig
    """Model used to generate agent responses during dialog evaluation."""

    reflection_provider: ModelProviderConfig | None = None
    """Mutation LM. If null, reuses task_provider."""

    mutator: MutatorConfig = Field(default_factory=MutatorConfig)

    train: list[Path] = Field(default_factory=lambda: [Path("datasets/augmented_train.json")])
    val: list[Path] = Field(default_factory=lambda: [Path("datasets/augmented_val.json")])
    test: list[Path] = Field(default_factory=list)
    """Optional held-out test set(s) for final evaluation of base vs best prompt."""
    seed_prompt: Path | None = None
    """Single-component seed prompt (backward compat for unblock/kib)."""
    seed_prompts: dict[str, Path] | None = None
    """Multi-component seed prompt (vis). Keys are component names, values are paths."""
    module_selector: Literal["round_robin", "all"] = "round_robin"
    """Component selection strategy passed to gepa.optimize()."""
    heatmap_template: Path | None = None
    """Optional path to Excel heatmap template. If None, a standalone heatmap is generated."""

    @field_validator("train", "val", "test", mode="before")
    @classmethod
    def coerce_to_list(cls, v: object) -> list:
        """Accept a single path string, a list of path strings, or None (→ [])."""
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            return [v]
        return v

    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    run_dir: Path = Path("artifacts/gepa_pipeline")
    """Results land in {run_dir}/{run_name}/."""

    @model_validator(mode="after")
    def validate_seed(self) -> "PipelineConfig":
        """Exactly one of seed_prompt / seed_prompts must be set."""
        has_single = self.seed_prompt is not None
        has_multi = self.seed_prompts is not None
        if has_single == has_multi:
            raise ValueError("Exactly one of 'seed_prompt' or 'seed_prompts' must be set.")
        return self

    @model_validator(mode="after")
    def resolve_paths(self) -> "PipelineConfig":
        root = Path(__file__).parent
        if self.seed_prompt is not None and not self.seed_prompt.is_absolute():
            object.__setattr__(self, "seed_prompt", (root / self.seed_prompt).resolve())
        if self.seed_prompts is not None:
            object.__setattr__(
                self, "seed_prompts",
                {
                    name: (root / p).resolve() if not p.is_absolute() else p
                    for name, p in self.seed_prompts.items()
                },
            )
        if not self.run_dir.is_absolute():
            object.__setattr__(self, "run_dir", (root / self.run_dir).resolve())
        if self.heatmap_template is not None and not self.heatmap_template.is_absolute():
            object.__setattr__(self, "heatmap_template", (root / self.heatmap_template).resolve())
        object.__setattr__(
            self, "train",
            [(root / p).resolve() if not p.is_absolute() else p for p in self.train],
        )
        object.__setattr__(
            self, "val",
            [(root / p).resolve() if not p.is_absolute() else p for p in self.val],
        )
        object.__setattr__(
            self, "test",
            [(root / p).resolve() if not p.is_absolute() else p for p in self.test],
        )
        return self

    @property
    def effective_reflection_provider(self) -> ModelProviderConfig:
        return self.reflection_provider or self.task_provider

    @property
    def effective_run_dir(self) -> Path:
        return self.run_dir / self.run_name


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    raw: dict = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(raw)
