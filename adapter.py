"""GEPAAdapter bridge between gepa.optimize() and our dialog pipeline."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypedDict

_log = logging.getLogger(__name__)

from gepa.core.adapter import EvaluationBatch

from llm import create_adapter, GenerationRequest, ModelProviderConfig


class StepTrace(TypedDict):
    step: int
    role: str
    action_type: str
    expected_message: str
    expected_pattern: str
    actual_response: str
    dialog_history: list[dict[str, str]]
    passed: bool


class DialogTrajectory(TypedDict):
    scenario_id: str
    scenario_description: str
    dialog_id: str
    steps_trace: list[StepTrace]


class DialogResult(TypedDict):
    scenario_id: str
    dialog_id: str
    steps_total: int
    steps_passed: int
    score: float


class UnblockCardAdapter:
    """GEPAAdapter that replays card-unblocking dialogs and validates with regex."""

    # Required by gepa GEPAAdapter protocol: reflective_mutation.py accesses
    # this attribute directly (not via getattr), so it must be declared.
    propose_new_texts = None

    def __init__(self, provider_config: ModelProviderConfig, temperature: float = 0.0, top_p: float | None = None) -> None:
        self._adapter = create_adapter(provider_config, top_p=top_p, temperature=temperature)
        self._temperature = temperature
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def _evaluate_scenario(
        self,
        scenario: dict[str, Any],
        system_prompt: str,
        capture_traces: bool,
        rollout_mode: str,
    ) -> tuple[DialogResult, float, dict[str, float], DialogTrajectory | None, int, int]:
        """Evaluate a single scenario. Thread-safe: does not mutate self."""
        dialogs = (
            [scenario] if "dialog_id" in scenario else scenario.get("dialogs", [])
        )

        steps_total = 0
        steps_passed = 0
        in_tokens = 0
        out_tokens = 0
        trajectory: DialogTrajectory | None = None

        if capture_traces:
            trajectory = DialogTrajectory(
                scenario_id=scenario.get("scenario_id", ""),
                scenario_description=scenario.get("scenario_description", ""),
                dialog_id=dialogs[0].get("dialog_id", "") if dialogs else "",
                steps_trace=[],
            )

        for dialog in dialogs:
            history: list[dict[str, str]] = []

            for step in dialog.get("steps", []):
                role = step.get("role")
                if role == "user":
                    history.append({"role": "user", "content": step["message"]})
                    continue
                if role != "agent":
                    continue

                response = self._adapter.generate(
                    GenerationRequest(
                        system_prompt=system_prompt,
                        history=history.copy(),
                        temperature=self._temperature,
                        metadata={
                            "scenario_id": scenario.get("scenario_id"),
                            "dialog_id": dialog.get("dialog_id"),
                            "step": step.get("step"),
                        },
                    )
                )
                if response.usage:
                    in_tokens += response.usage.input_tokens
                    out_tokens += response.usage.output_tokens

                pattern = step.get("additional_check", "")
                if pattern:
                    try:
                        passed = bool(re.search(pattern, response.output_text, re.S | re.I))
                    except re.error as exc:
                        _log.warning(
                            "Invalid regex %r in scenario %s step %s: %s — treating as failed",
                            pattern, scenario.get("scenario_id"), step.get("step"), exc,
                        )
                        passed = False
                else:
                    passed = True

                steps_total += 1
                if passed:
                    steps_passed += 1

                if capture_traces and trajectory is not None:
                    trajectory["steps_trace"].append(
                        StepTrace(
                            step=step.get("step", 0),
                            role=role,
                            action_type=step.get("type", ""),
                            expected_message=step.get("message", ""),
                            expected_pattern=pattern,
                            actual_response=response.output_text,
                            dialog_history=history.copy(),
                            passed=passed,
                        )
                    )

                if rollout_mode == "real_rollout":
                    history.append({"role": "assistant", "content": response.output_text})
                else:
                    history.append({"role": "assistant", "content": step["message"]})

        score = steps_passed / steps_total if steps_total > 0 else 0.0

        result = DialogResult(
            scenario_id=scenario.get("scenario_id", ""),
            dialog_id=dialogs[0].get("dialog_id", "") if dialogs else "",
            steps_total=steps_total,
            steps_passed=steps_passed,
            score=score,
        )
        return result, score, {"accuracy": score}, trajectory, in_tokens, out_tokens

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
        rollout_mode: Literal["golden_path", "real_rollout"] = "golden_path",
    ) -> EvaluationBatch:
        system_prompt = candidate["system_prompt"]
        max_workers = min(len(batch), 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_scenario, scenario, system_prompt,
                    capture_traces, rollout_mode,
                ): i
                for i, scenario in enumerate(batch)
            }
            indexed: list[tuple[int, tuple]] = []
            for future in as_completed(futures):
                indexed.append((futures[future], future.result()))

        indexed.sort(key=lambda x: x[0])

        outputs: list[DialogResult] = []
        scores: list[float] = []
        trajectories: list[DialogTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []

        for _, (result, score, obj, traj, in_tok, out_tok) in indexed:
            outputs.append(result)
            scores.append(score)
            objective_scores.append(obj)
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            if capture_traces and traj is not None:
                trajectories.append(traj)  # type: ignore[union-attr]

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
        components_to_update: list[str],  # unused: adapter optimises system_prompt only; required by GEPAAdapter protocol
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {"system_prompt": []}

        records: list[Mapping[str, Any]] = []

        for trajectory in eval_batch.trajectories:
            steps = trajectory["steps_trace"]
            steps_total = len(steps)
            steps_passed = sum(1 for s in steps if s["passed"])

            for step_trace in steps:
                if step_trace["passed"]:
                    continue

                history_json = "\n".join(
                    f"  {i}: {msg}"
                    for i, msg in enumerate(step_trace["dialog_history"])
                )

                records.append(
                    {
                        "Inputs": {
                            "dialog_history": f"```json\n{history_json}\n```",
                            "expected_pattern": step_trace["expected_pattern"],
                        },
                        "Generated Outputs": step_trace["actual_response"],
                        "Feedback": (
                            f"Шаг {step_trace['step']}: ответ не соответствует паттерну. "
                            f"Ожидалось: {step_trace['expected_message']}. "
                            f"Паттерн: {step_trace['expected_pattern']}. "
                            f"Сценарий: {steps_passed}/{steps_total} шагов пройдено."
                        ),
                    }
                )

        return {"system_prompt": records}
