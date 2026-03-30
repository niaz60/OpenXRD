from __future__ import annotations

import json

from openxrd.evaluator import evaluate


def _write_sample_dataset(data_dir):
    questions = [
        {
            "question": "How many crystal systems are there in crystallography?",
            "options": ["5", "6", "7", "8"],
            "correct_answer": 2,
            "category": "Crystallography",
            "subtask": "Structural_Analysis",
        },
        {
            "question": "Which law relates scattering angle and wavelength?",
            "options": ["Snell's law", "Bragg's law", "Hooke's law", "Boyle's law"],
            "correct_answer": 1,
            "category": "XRD",
            "subtask": "Diffraction",
        },
    ]
    materials = [
        {"question_id": 0, "helper_text": "There are seven crystal systems."},
        {"question_id": 1, "helper_text": "Bragg's law is used in X-ray diffraction."},
    ]

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "benchmarking_questions.json").write_text(json.dumps(questions), encoding="utf-8")
    (data_dir / "supporting_textual_materials_generated.json").write_text(
        json.dumps(materials),
        encoding="utf-8",
    )
    (data_dir / "supporting_textual_materials_expert_reviewed.json").write_text(
        json.dumps(materials),
        encoding="utf-8",
    )


def test_evaluate_with_mocked_provider(monkeypatch, tmp_path):
    data_dir = tmp_path / "openxrd"
    _write_sample_dataset(data_dir)
    monkeypatch.setenv("OPENXRD_DATA_DIR", str(data_dir))

    responses = iter(["3", "2"])

    def fake_caller(_prompt: str, _system_prompt: str) -> str:
        return next(responses)

    monkeypatch.setattr("openxrd.evaluator.make_model_caller", lambda provider, model: fake_caller)

    results = evaluate(
        provider="openai",
        model="dummy-model",
        mode="openbook",
        limit=2,
        material_kind="expert_reviewed",
    )

    assert results["total_questions"] == 2
    assert results["correct_answers"] == 2
    assert results["accuracy"] == 1.0
    assert results["results"][0]["predicted_option_number"] == 3
