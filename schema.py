"""Pydantic models for dataset validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StepData(BaseModel):
    step: int | None = None
    role: Literal["user", "agent"]
    message: str = Field(min_length=1)
    additional_check: str | None = None


class DialogData(BaseModel):
    dialog_id: str | None = None
    steps: list[StepData] = Field(min_length=1)


class ScenarioData(BaseModel):
    scenario_id: str | None = None
    dialog_id: str | None = None
    steps: list[StepData] | None = None
    dialogs: list[DialogData] | None = None

    @field_validator("steps", "dialogs", mode="after")
    @classmethod
    def at_least_one_present(cls, v, info):
        return v

    def model_post_init(self, __context) -> None:
        # A scenario must have either inline steps (with dialog_id) or a dialogs list.
        has_inline = self.dialog_id is not None and self.steps is not None and len(self.steps) > 0
        has_dialogs = self.dialogs is not None and len(self.dialogs) > 0
        if not has_inline and not has_dialogs:
            raise ValueError("scenario must have either (dialog_id + steps) or non-empty dialogs list")


def load_and_validate_dataset(paths: list[Path]) -> list[dict]:
    """Load JSON dataset files, validate with Pydantic, return plain dicts."""
    import json

    result: list[dict] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to read dataset file {path}: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError(f"Dataset file {path} must contain a JSON array, got {type(raw).__name__}")

        for idx, entry in enumerate(raw):
            try:
                ScenarioData.model_validate(entry)
            except Exception as exc:
                raise ValueError(f"Validation error in {path} entry #{idx}: {exc}") from exc
            result.append(entry)

    return result
