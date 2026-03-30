"""Dataset location and loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .constants import (
    DATA_ARCHIVE_NAME,
    DEFAULT_DATA_SUBDIR,
    EXPERT_FILE,
    GENERATED_FILE,
    QUESTIONS_FILE,
)

Question = Dict[str, Any]
Material = Dict[str, Any]


def get_repo_root() -> Path:
    """Return the repository root for an editable repo-first install."""

    return Path(__file__).resolve().parents[2]


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the extracted dataset directory."""

    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    override = os.getenv("OPENXRD_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    return (get_repo_root() / DEFAULT_DATA_SUBDIR).resolve()


def get_dataset_archive_path() -> Path:
    """Return the tracked zip archive path."""

    return get_repo_root() / "datasets" / DATA_ARCHIVE_NAME


def _required_data_files(data_dir: Path) -> Dict[str, Path]:
    return {
        QUESTIONS_FILE: data_dir / QUESTIONS_FILE,
        GENERATED_FILE: data_dir / GENERATED_FILE,
        EXPERT_FILE: data_dir / EXPERT_FILE,
    }


def get_data_status(data_dir: str | Path | None = None) -> Dict[str, Any]:
    """Return archive and extracted-data status for CLI reporting."""

    resolved_dir = get_data_dir(data_dir)
    expected = _required_data_files(resolved_dir)
    missing = [name for name, path in expected.items() if not path.exists()]

    return {
        "repo_root": get_repo_root(),
        "archive_path": get_dataset_archive_path(),
        "archive_exists": get_dataset_archive_path().exists(),
        "data_dir": resolved_dir,
        "expected_files": expected,
        "missing_files": missing,
        "data_ready": not missing,
    }


def _missing_data_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"Missing extracted dataset file: {path}. "
        "Run `./scripts/unzip_dataset.sh --acknowledge` from the repository root first."
    )


def _read_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise _missing_data_error(path) from exc

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, found {type(payload).__name__}")
    return payload


def load_questions(data_dir: str | Path | None = None) -> List[Question]:
    """Load the extracted OpenXRD benchmark questions."""

    path = get_data_dir(data_dir) / QUESTIONS_FILE
    return _read_json(path)


def load_supporting_materials(
    kind: str = "expert_reviewed",
    data_dir: str | Path | None = None,
) -> List[Material]:
    """Load extracted supporting materials."""

    if kind not in {"expert_reviewed", "generated"}:
        raise ValueError("kind must be 'expert_reviewed' or 'generated'")

    filename = EXPERT_FILE if kind == "expert_reviewed" else GENERATED_FILE
    return _read_json(get_data_dir(data_dir) / filename)
