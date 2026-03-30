from __future__ import annotations

import json

from openxrd.cli import main_check, main_example


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


def test_main_check_reports_missing_data(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENXRD_DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert main_check([]) == 0
    output = capsys.readouterr().out

    assert "Extracted dataset: missing files" in output
    assert "OPENAI_API_KEY -> missing" in output
    assert "OPENROUTER_API_KEY -> missing" in output


def test_main_example_supports_openai_and_openrouter(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "openxrd"
    _write_sample_dataset(data_dir)
    monkeypatch.setenv("OPENXRD_DATA_DIR", str(data_dir))

    def run_case(provider: str):
        responses = iter(["3", "2"])

        def fake_caller(_prompt: str, _system_prompt: str) -> str:
            return next(responses)

        monkeypatch.setattr("openxrd.evaluator.make_model_caller", lambda _provider, _model: fake_caller)
        assert main_example(["--provider", provider, "--model", "dummy-model", "--limit", "2"]) == 0

    run_case("openai")
    openai_output = capsys.readouterr().out
    assert "openai:dummy-model" in openai_output

    run_case("openrouter")
    openrouter_output = capsys.readouterr().out
    assert "openrouter:dummy-model" in openrouter_output
