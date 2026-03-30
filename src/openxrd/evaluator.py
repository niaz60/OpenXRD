"""Evaluation logic for the extracted OpenXRD dataset."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .data import load_questions, load_supporting_materials
from .providers import make_model_caller

SYSTEM_PROMPT = (
    "You are a careful assistant answering multiple-choice X-ray diffraction and "
    "crystallography questions. Reply with only the number of the best answer."
)


def format_prompt(
    question: str,
    options: List[str],
    supporting_material: str | None = None,
) -> str:
    """Create the prompt sent to the model."""

    rendered_options = "\n".join(f"{index + 1}. {option}" for index, option in enumerate(options))
    if supporting_material:
        return (
            "Reference material:\n"
            f"{supporting_material}\n\n"
            "Use the reference material to answer the question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{rendered_options}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)})."
        )

    return (
        f"Question: {question}\n\n"
        f"Options:\n{rendered_options}\n\n"
        f"Reply with only the number of the correct option (1-{len(options)})."
    )


def parse_model_response(response: str, num_options: int) -> int:
    """Extract a zero-based answer index from a model response."""

    for token in re.findall(r"\d+", response):
        value = int(token)
        if 1 <= value <= num_options:
            return value - 1
    return 0


def evaluate(
    provider: str,
    model: str,
    mode: str = "closedbook",
    limit: int | None = None,
    material_kind: str = "expert_reviewed",
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run a benchmark evaluation against the extracted dataset."""

    if mode not in {"closedbook", "openbook"}:
        raise ValueError("mode must be 'closedbook' or 'openbook'")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    questions = load_questions(data_dir=data_dir)
    if limit is not None:
        questions = questions[:limit]

    supporting_materials = None
    if mode == "openbook":
        supporting_materials = load_supporting_materials(kind=material_kind, data_dir=data_dir)
        if limit is not None:
            supporting_materials = supporting_materials[:limit]
        if len(supporting_materials) < len(questions):
            raise ValueError("Not enough supporting materials for the requested evaluation.")

    caller = make_model_caller(provider, model)
    results: List[Dict[str, Any]] = []

    for index, question in enumerate(questions):
        supporting_material = None
        if supporting_materials is not None:
            supporting_material = supporting_materials[index].get("helper_text", "")

        prompt = format_prompt(
            question=question["question"],
            options=question["options"],
            supporting_material=supporting_material,
        )
        raw_response = caller(prompt, SYSTEM_PROMPT)
        predicted_index = parse_model_response(raw_response, len(question["options"]))
        correct_index = int(question["correct_answer"])
        is_correct = predicted_index == correct_index

        results.append(
            {
                "question_id": index,
                "question": question["question"],
                "options": question["options"],
                "predicted_index": predicted_index,
                "predicted_option_number": predicted_index + 1,
                "correct_answer": correct_index,
                "correct_option_number": correct_index + 1,
                "is_correct": is_correct,
                "category": question.get("category", "Unknown"),
                "subtask": question.get("subtask", "Unknown"),
                "raw_response": raw_response,
            }
        )

    payload: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "mode": mode,
        "material_kind": material_kind if mode == "openbook" else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "correct_answers": sum(1 for result in results if result["is_correct"]),
        "accuracy": 0.0,
        "category_metrics": _group_metrics(results, "category"),
        "subtask_metrics": _group_metrics(results, "subtask"),
        "results": results,
    }
    if payload["total_questions"]:
        payload["accuracy"] = payload["correct_answers"] / payload["total_questions"]

    if output_path is not None:
        _write_results(payload, output_path)

    return payload


def _group_metrics(results: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        key = str(result.get(field, "Unknown"))
        bucket = grouped.setdefault(key, {"count": 0, "correct": 0, "accuracy": 0.0})
        bucket["count"] += 1
        if result["is_correct"]:
            bucket["correct"] += 1

    for bucket in grouped.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else 0.0

    return dict(sorted(grouped.items()))


def _write_results(payload: Dict[str, Any], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
