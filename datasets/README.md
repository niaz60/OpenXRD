# OpenXRD Dataset Archive

This directory intentionally exposes the benchmark as a zip archive:

- `openxrd_dataset.zip`

The archive contains:

- `benchmarking_questions.json`
- `supporting_textual_materials_generated.json`
- `supporting_textual_materials_expert_reviewed.json`
- `README.md`

## Why The Dataset Is Zipped

The dataset is zipped to add friction against casual scraping and to reduce accidental ingestion by online aggregation or crawling tools. This is not strong protection or access control.

## Usage Policy

- OpenXRD is an evaluation-only dataset.
- The dataset must not be used to train, fine-tune, distill, align, or otherwise optimize machine learning models.
- Use of the dataset is conditioned on citing the accepted OpenXRD paper in `Digital Discovery`.
- Extract it with `./scripts/unzip_dataset.sh --acknowledge`.
