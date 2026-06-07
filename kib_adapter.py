"""GEPAAdapter for КИБ LLM-filter classification with Structured Output."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from gepa.core.adapter import EvaluationBatch

from llm import create_adapter, GenerationRequest, ModelProviderConfig

_log = logging.getLogger(__name__)

_VALID_LABELS = frozenset({"off_topic", "negative", "no_capability", "in_topic"})

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "GradeInputQuery",
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": sorted(_VALID_LABELS),
                }
            },
            "required": ["label"],
        },
    },
}


class StepTrace(TypedDict):
    step: int
    role: str
    expected_label: str
    actual_label: str
    actual_raw: str
    user_message: str
    passed: bool


class DialogTrajectory(TypedDict):
    scenario_id: str
    dialog_id: str
    steps_trace: list[StepTrace]


class DialogResult(TypedDict):
    scenario_id: str
    dialog_id: str
    steps_total: int
    steps_passed: int
    score: float


def _extract_label(text: str) -> str | None:
    """Try to extract a valid label from model output (JSON or plain text)."""
    # Try JSON parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            lbl = obj.get("label", "").strip().lower()
            if lbl in _VALID_LABELS:
                return lbl
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: look for a known label as a standalone word
    text_lower = text.strip().lower()
    for label in _VALID_LABELS:
        if re.search(rf"\b{label}\b", text_lower):
            return label

    return None


class KIBAdapter:
    """GEPAAdapter for КИБ single-turn classification with Structured Output."""

    propose_new_texts = None  # GEPAAdapter protocol

    def __init__(self, provider_config: ModelProviderConfig, temperature: float = 0.0, top_p: float | None = None) -> None:
        self._adapter = create_adapter(provider_config, top_p=top_p, temperature=temperature)
        self._temperature = temperature
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
        rollout_mode: Literal["golden_path", "real_rollout"] = "golden_path",
    ) -> EvaluationBatch:
        system_prompt = candidate["system_prompt"]

        outputs: list[DialogResult] = []
        scores: list[float] = []
        trajectories: list[DialogTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []

        for scenario in batch:
            steps = scenario.get("steps", [])

            # Extract user message (first user step) and expected label (agent step)
            user_message = ""
            expected_label = ""
            expected_pattern = ""

            for step in steps:
                if step["role"] == "user":
                    user_message = step["message"]
                elif step["role"] == "agent":
                    expected_label = step["message"]
                    expected_pattern = step.get("additional_check", "")

            # Generate classification
            response = self._adapter.generate(
                GenerationRequest(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=self._temperature,
                    response_format=_RESPONSE_FORMAT,
                    metadata={
                        "scenario_id": scenario.get("scenario_id"),
                        "dialog_id": scenario.get("dialog_id"),
                    },
                )
            )
            if response.usage:
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens

            # Extract label from response
            actual_label = _extract_label(response.output_text)

            # Check correctness
            if expected_pattern:
                try:
                    passed = bool(re.search(expected_pattern, actual_label or "", re.S | re.I))
                except re.error as exc:
                    _log.warning(
                        "Invalid regex %r in scenario %s: %s — treating as failed",
                        expected_pattern, scenario.get("scenario_id"), exc,
                    )
                    passed = False
            else:
                passed = (actual_label == expected_label)

            score = 1.0 if passed else 0.0

            outputs.append(
                DialogResult(
                    scenario_id=scenario.get("scenario_id", ""),
                    dialog_id=scenario.get("dialog_id", ""),
                    steps_total=1,
                    steps_passed=1 if passed else 0,
                    score=score,
                )
            )
            scores.append(score)
            objective_scores.append({"accuracy": score})

            if capture_traces and trajectories is not None:
                trajectories.append(
                    DialogTrajectory(
                        scenario_id=scenario.get("scenario_id", ""),
                        dialog_id=scenario.get("dialog_id", ""),
                        steps_trace=[
                            StepTrace(
                                step=1,
                                role="agent",
                                expected_label=expected_label,
                                actual_label=actual_label or "",
                                actual_raw=response.output_text,
                                user_message=user_message,
                                passed=passed,
                            )
                        ],
                    )
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {"system_prompt": []}

        records: list[Mapping[str, Any]] = []

        for trajectory in eval_batch.trajectories:
            for step_trace in trajectory["steps_trace"]:
                if step_trace["passed"]:
                    continue

                records.append(
                    {
                        "Inputs": {
                            "dialog_history": f"User: {step_trace['user_message']}",
                            "expected_pattern": f"^{step_trace['expected_label']}$",
                        },
                        "Generated Outputs": step_trace["actual_raw"],
                        "Feedback": (
                            f"Классификация неверна. "
                            f"Ожидалось: {step_trace['expected_label']}. "
                            f"Получено: {step_trace['actual_label']}. "
                            f"Запрос клиента: {step_trace['user_message']}"
                        ),
                    }
                )

        return {"system_prompt": records}
