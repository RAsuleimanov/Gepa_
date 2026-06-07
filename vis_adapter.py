"""VisAdapter — enriched feedback for ВиС task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import EvaluationBatch

from adapter import UnblockCardAdapter


class VisAdapter(UnblockCardAdapter):
    """UnblockCardAdapter with enriched, human-readable feedback for ВиС."""

    def evaluate(self, batch, candidate, **kwargs):
        """Combine multi-component candidate into a single system_prompt for evaluation."""
        if "system_prompt" not in candidate:
            combined = {"system_prompt": "\n\n".join(candidate.values())}
        else:
            combined = candidate
        return super().evaluate(batch, combined, **kwargs)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {name: [] for name in components_to_update}

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

                pattern = step_trace.get("expected_pattern", "")
                feedback_parts = [
                    f"Тип ответа: {step_trace['action_type']}.",
                    f"Получено: '{step_trace['actual_response']}'.",
                    f"Ожидалось: '{step_trace['expected_message']}'.",
                ]
                if pattern:
                    feedback_parts.append(
                        f"Regex-паттерн проверки: {pattern}"
                    )
                feedback_parts.append(
                    f"Сценарий: {steps_passed}/{steps_total} шагов пройдено."
                )

                records.append(
                    {
                        "Inputs": {
                            "dialog_history": f"```json\n{history_json}\n```",
                        },
                        "Generated Outputs": step_trace["actual_response"],
                        "Feedback": " ".join(feedback_parts),
                    }
                )

        # For each component being updated, include the OTHER components' text
        # and strict boundary rules to prevent content duplication.
        result: dict[str, list[Mapping[str, Any]]] = {}
        for name in components_to_update:
            other_parts = {
                k: v for k, v in candidate.items() if k != name
            }
            if other_parts:
                context_record = {
                    "Inputs": {
                        "context": self._build_context(name, other_parts),
                    },
                    "Generated Outputs": "",
                    "Feedback": "Это контекст, а не ошибка.",
                }
                result[name] = [context_record] + list(records)
            else:
                result[name] = list(records)

        return result

    @staticmethod
    def _build_context(component_name: str, other_parts: dict[str, str]) -> str:
        other_text = "\n\n".join(f"[{k}]:\n{v}" for k, v in other_parts.items())

        if component_name == "routing_logic":
            return (
                "ВАЖНО: Ты редактируешь ТОЛЬКО компонент routing_logic.\n"
                "routing_logic отвечает за: алгоритм принятия решений, приоритеты, формат ответа, "
                "правила распознавания намерений, особые случаи маршрутизации.\n"
                "routing_logic НЕ должен содержать: перечень документов, номера документов, "
                "примечания к документам, примеры запросов к конкретным документам.\n"
                "Перечень документов находится в другом компоненте (doc_catalog), "
                "не дублируй его содержимое.\n\n"
                f"Содержимое doc_catalog (только для справки, не копируй):\n{other_text}"
            )

        if component_name == "doc_catalog":
            return (
                "ВАЖНО: Ты редактируешь ТОЛЬКО компонент doc_catalog.\n"
                "doc_catalog отвечает за: перечень документов с номерами и названиями, "
                "примечания к документам, примеры запросов.\n"
                "doc_catalog НЕ должен содержать: алгоритм принятия решений, приоритеты, "
                "формат ответа (FAQ/ОПЕРАТОР/КЛИЕНТ), правила маршрутизации, "
                "правила безопасности.\n"
                "Правила маршрутизации находятся в другом компоненте (routing_logic), "
                "не дублируй его содержимое.\n\n"
                f"Содержимое routing_logic (только для справки, не копируй):\n{other_text}"
            )

        # Fallback for unknown components.
        return (
            f"Ты редактируешь ТОЛЬКО компонент {component_name}. "
            "Не дублируй содержимое других компонентов.\n\n"
            f"Другие компоненты (только для справки):\n{other_text}"
        )
