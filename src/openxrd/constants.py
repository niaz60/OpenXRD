"""Shared constants for the OpenXRD package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_NAME = "openxrd"
PACKAGE_VERSION = "1.0.0"

DATA_ARCHIVE_NAME = "openxrd_dataset.zip"
DEFAULT_DATA_SUBDIR = Path("data") / "openxrd"

QUESTIONS_FILE = "benchmarking_questions.json"
GENERATED_FILE = "supporting_textual_materials_generated.json"
EXPERT_FILE = "supporting_textual_materials_expert_reviewed.json"

PAPER_TITLE = "OPENXRD: A Comprehensive Benchmark Framework for LLM/MLLM XRD Question Answering"
PAPER_DOI = "10.1039/D5DD00519A"
PAPER_URL = f"https://doi.org/{PAPER_DOI}"
PAPER_CITATION = (
    "A. Vosoughi, A. Shahnazari, Z. Zhang, Y. Xi, G. Hess, C. Xu and "
    "N. Abdolrahim, Digital Discovery, 2025, Accepted Manuscript, "
    f"DOI: {PAPER_DOI}"
)

DATA_POLICY_LINES = [
    "OpenXRD is an evaluation-only dataset.",
    "It must not be used to train, fine-tune, distill, align, or otherwise optimize models.",
    "Use of the dataset is conditioned on citing the accepted Digital Discovery paper.",
]

ZIP_RATIONALE = (
    "The dataset is distributed as a zip archive to reduce casual scraping and "
    "accidental ingestion by online aggregation or crawling tools. This is friction, "
    "not strong protection or access control."
)
