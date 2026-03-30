# OpenXRD

[![Paper](https://img.shields.io/badge/Paper-Digital%20Discovery-blue.svg)](https://pubs.rsc.org/en/Content/ArticleLanding/2025/DD/D5DD00519A)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD5DD00519A-1f6feb.svg)](https://doi.org/10.1039/D5DD00519A)
[![Website](https://img.shields.io/badge/Website-OpenXRD-blue)](https://niaz60.github.io/OpenXRD/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green.svg)](LICENSE)

OpenXRD is a repo-first, researcher-facing package for evaluating language models on the OpenXRD crystallography benchmark. This refresh keeps the public release intentionally small:

1. install from the repo,
2. acknowledge the dataset policy,
3. unzip the evaluation dataset,
4. set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`,
5. run a short evaluation example.

The accepted paper is published in *Digital Discovery* with DOI [`10.1039/D5DD00519A`](https://doi.org/10.1039/D5DD00519A). RSC lists the manuscript as accepted on March 9, 2026 and first published on March 16, 2026.

## Install

### Recommended

```bash
git clone https://github.com/niaz60/OpenXRD.git
cd OpenXRD
./scripts/install.sh
source .venv/bin/activate
```

`./scripts/install.sh` prefers `uv` when available and falls back to `python -m venv` plus `pip` otherwise.

Recommended and tested setup: Python `3.13`. The package metadata currently allows `3.9+`, but `3.13` is the documented target for the most reproducible local install.

### Direct Alternative

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `python3.13` is not on your `PATH`, use:

```bash
PYTHON_BIN=python3.13 ./scripts/install.sh
```

## Dataset

The public repo tracks the dataset as [`datasets/openxrd_dataset.zip`](datasets/openxrd_dataset.zip), not as extracted JSON files.

Why the dataset is zipped:

- It adds friction against casual scraping.
- It reduces accidental ingestion by online aggregation and crawling tools.
- It is not strong protection or access control.

OpenXRD is an evaluation-only dataset:

- It must not be used to train, fine-tune, distill, align, or otherwise optimize models.
- Use of the dataset is conditioned on citing the accepted *Digital Discovery* paper.

Extract the dataset only after acknowledging that policy:

```bash
./scripts/unzip_dataset.sh --acknowledge
```

By default this extracts to `data/openxrd/`.

## Quick Start

### Check Your Setup

```bash
openxrd-check
```

This reports:

- whether the dataset archive is present,
- whether extracted dataset files are available,
- whether `OPENAI_API_KEY` and `OPENROUTER_API_KEY` are set.

### OpenAI Example

Create an OpenAI API key at `https://platform.openai.com/api-keys`, then:

```bash
export OPENAI_API_KEY="your-openai-key"
openxrd-example --provider openai --model your-openai-model --limit 3
```

### OpenRouter Example

Create an OpenRouter API key at `https://openrouter.ai/keys`, then:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
openxrd-example --provider openrouter --model anthropic/claude-3.5-sonnet --limit 3
```

OpenRouter can be used to evaluate Claude-family models through the same OpenAI-compatible interface.

### Python Example

```python
from openxrd import evaluate

results = evaluate(
    provider="openai",
    model="your-openai-model",
    mode="closedbook",
    limit=3,
)

print(results["accuracy"])
```

A short runnable example is also available at [`examples/basic_usage.py`](examples/basic_usage.py).

## Public API

The public package intentionally exposes a small API:

- `load_questions()`
- `load_supporting_materials(kind="expert_reviewed" | "generated")`
- `evaluate(provider, model, mode, limit, material_kind, output_path=None)`

Supported providers in this public release:

- `openai`
- `openrouter`

## Citation

If you use OpenXRD, cite:

```bibtex
@article{Vosoughi_2025,
  title = {OPENXRD: A Comprehensive Benchmark Framework for LLM/MLLM XRD Question Answering},
  author = {Vosoughi, Ali and Shahnazari, Ayoub and Zhang, Zeliang and Xi, Yufeng and Hess, Griffin and Xu, Chenliang and Abdolrahim, Niaz},
  journal = {Digital Discovery},
  publisher = {Royal Society of Chemistry (RSC)},
  year = {2025},
  doi = {10.1039/D5DD00519A},
  url = {https://doi.org/10.1039/D5DD00519A}
}
```

RSC citation string:

`A. Vosoughi, A. Shahnazari, Z. Zhang, Y. Xi, G. Hess, C. Xu and N. Abdolrahim, Digital Discovery, 2025, Accepted Manuscript, DOI: 10.1039/D5DD00519A`

## Development

For local verification:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Website Publishing

The project website at `https://niaz60.github.io/OpenXRD/` is published from `main` via GitHub Actions.

- The source of truth stays at the repo root in `index.html`.
- The Pages deployment workflow is defined in `.github/workflows/pages.yml`.
- In GitHub repository settings, Pages should be configured to use `GitHub Actions` as the source.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. See [LICENSE](LICENSE).
