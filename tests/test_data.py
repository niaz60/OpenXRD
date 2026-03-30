from __future__ import annotations

import json

import pytest

from openxrd.data import load_questions, load_supporting_materials


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


def test_loaders_require_extracted_data(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENXRD_DATA_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="unzip_dataset.sh --acknowledge"):
        load_questions()


def test_loaders_read_extracted_data(monkeypatch, tmp_path):
    data_dir = tmp_path / "openxrd"
    _write_sample_dataset(data_dir)
    monkeypatch.setenv("OPENXRD_DATA_DIR", str(data_dir))

    questions = load_questions()
    materials = load_supporting_materials()

    assert len(questions) == 2
    assert questions[0]["correct_answer"] == 2
    assert materials[0]["helper_text"].startswith("There are seven")
