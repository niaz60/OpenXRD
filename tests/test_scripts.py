from __future__ import annotations

import subprocess
from pathlib import Path


def test_unzip_script_requires_acknowledgement(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "openxrd"

    result = subprocess.run(
        ["bash", "scripts/unzip_dataset.sh", "--output-dir", str(output_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing to extract without explicit acknowledgment" in result.stderr
    assert not output_dir.exists()


def test_unzip_script_extracts_dataset(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "openxrd"

    result = subprocess.run(
        [
            "bash",
            "scripts/unzip_dataset.sh",
            "--acknowledge",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "benchmarking_questions.json").exists()
    assert (output_dir / "supporting_textual_materials_generated.json").exists()
    assert (output_dir / "supporting_textual_materials_expert_reviewed.json").exists()
