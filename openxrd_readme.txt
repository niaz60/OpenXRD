# OpenXRD: A Comprehensive Benchmark and Enhancement Framework for LLM/MLLM XRD Question Answering

[![Paper](https://img.shields.io/badge/arXiv-2507.09155-b31b1b.svg)](https://arxiv.org/abs/2507.09155)
[![Website](https://img.shields.io/badge/Website-OpenXRD-blue)](https://niaz60.github.io/OpenXRD/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation Scripts](#evaluation-scripts)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## 🔬 Overview

OpenXRD presents an open-book pipeline designed for crystallography question answering, integrating textual prompts with concise supporting content. Instead of using scanned textbooks (which may lead to copyright issues), OpenXRD generates compact, domain-specific references that help smaller models understand key concepts in X-ray diffraction (XRD).

### Key Findings
- **Mid-capacity models** (7B-34B parameters) benefit most from external knowledge
- **Up to +11.5%** improvement with expert-reviewed materials vs +6% with AI-generated content alone
- **Inverted U relationship**: Largest models show minimal improvement while smallest models have limited capacity to utilize additional information

## ✨ Features

- **Comprehensive Benchmark**: 217 expert-level XRD multiple-choice questions
- **Dual Evaluation Modes**: Closed-book vs Open-book evaluation framework
- **Multiple Model Support**: GPT-4, O1, O3, LLaVA, Gemini, and more
- **Expert-Reviewed Materials**: AI-generated supporting content refined by crystallography experts
- **Detailed Analysis**: Subtask-level performance analysis and visualization tools

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Required API keys (see Configuration section)

### Setup
```bash
git clone https://github.com/niaz60/OpenXRD.git
cd OpenXRD
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory:
```bash
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Google API Key (for Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face Token (for LLaVA models, optional)
HUGGINGFACE_TOKEN=your_hf_token_here
```

## 📊 Dataset

The repository includes three main dataset files in the `datasets/` directory:

- **`benchmarking_questions.json`**: 217 expert-curated XRD questions with explanations and subtask labels
- **`supporting_textual_materials_generated.json`**: AI-generated supporting materials for open-book evaluation
- **`supporting_textual_materials_expert_reviewed.json`**: Expert-refined supporting materials

### Dataset Structure
```json
{
  "question": "What happens if the path difference between scattered rays is nλ?",
  "options": ["They cancel each other", "They reinforce each other", "...", "..."],
  "correct_answer": 1,
  "explanation": "When two waves differ in path by nλ...",
  "category": "Wave Physics",
  "subtask": "Interference Phenomena"
}
```

## 💻 Usage

### Quick Start
```python
from src.evaluation import evaluate_model
from src.utils import load_dataset

# Load dataset
questions = load_dataset("datasets/benchmarking_questions.json")

# Run evaluation (example with OpenAI)
results = evaluate_model(
    model_type="openai",
    model_name="gpt-4",
    questions=questions,
    mode="closedbook"  # or "openbook"
)

print(f"Accuracy: {results['accuracy']:.2%}")
```

### Running Evaluations

#### OpenAI Models (GPT-4, O1, O3)
```bash
python scripts/evaluate_openai.py --model gpt-4 --mode closedbook
python scripts/evaluate_openai.py --model gpt-4 --mode openbook
```

#### Gemini Models
```bash
python scripts/evaluate_gemini.py --model gemini-2.0-flash --mode closedbook
```

#### LLaVA Models
```bash
python scripts/evaluate_llava.py --model llava-v1.6-34b --mode openbook
```

#### Batch Evaluation
```bash
python scripts/run_all_evaluations.py
```

## 📁 Evaluation Scripts

### Core Evaluation Scripts
- `scripts/evaluate_openai.py` - OpenAI models (GPT-4, O1, O3)
- `scripts/evaluate_gemini.py` - Google Gemini models
- `scripts/evaluate_llava.py` - LLaVA vision-language models
- `scripts/evaluate_llava_next.py` - LLaVA-NeXT models

### Analysis Scripts
- `scripts/analyze_subtasks.py` - Detailed subtask-level analysis
- `scripts/generate_wordcloud.py` - Visualization of subtask distribution
- `scripts/universal_subtask_analysis.py` - Cross-model comparison

### Utility Scripts
- `scripts/generate_docx.py` - Export questions to Word document
- `scripts/reasoner_cheater.py` - Generate supporting materials

## 📈 Results

### Model Performance (Closed-book)
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | GPT-4.5-preview | 93.09% |
| 2 | O3-mini | 88.94% |
| 3 | O1 | 87.56% |
| 4 | GPT-4-turbo | 83.41% |
| 5 | LLaVA-v1.6-34B | 66.80% |

### Improvement with Expert-Reviewed Materials
| Model | Closed-book | Open-book | Improvement |
|-------|-------------|-----------|-------------|
| LLaVA-v1.6-34B | 66.80% | 78.30% | +11.50% |
| LLaVA-v1.6-mistral-7B | 53.00% | 64.10% | +11.10% |
| O3-mini | 88.94% | 89.90% | +0.96% |

## 🔧 Project Structure

```
OpenXRD/
├── datasets/                          # Dataset files
│   ├── benchmarking_questions.json
│   ├── supporting_textual_materials_generated.json
│   └── supporting_textual_materials_expert_reviewed.json
├── src/                              # Core source code
│   ├── __init__.py
│   ├── evaluation.py                 # Main evaluation functions
│   ├── models/                       # Model-specific implementations
│   │   ├── openai_models.py
│   │   ├── gemini_models.py
│   │   └── llava_models.py
│   └── utils.py                      # Utility functions
├── scripts/                          # Evaluation scripts
│   ├── evaluate_openai.py
│   ├── evaluate_gemini.py
│   ├── evaluate_llava.py
│   ├── analyze_subtasks.py
│   └── run_all_evaluations.py
├── results/                          # Output directory
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

## 📚 Citation

If you use OpenXRD in your research, please cite our paper:

```bibtex
@article{vosoughi2025openxrd,
  title={OPENXRD: A Comprehensive Benchmark and Enhancement Framework for LLM/MLLM XRD Question Answering},
  author={Vosoughi, Ali and Shahnazari, Ayoub and Xi, Yufeng and Zhang, Zeliang and Hess, Griffin and Xu, Chenliang and Abdolrahim, Niaz},
  journal={arXiv preprint arXiv:2507.09155},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ali Vosoughi** - University of Rochester
- **Ayoub Shahnazari** - University of Rochester  
- **Yufeng Xi** - University of Rochester
- **Zeliang Zhang** - University of Rochester
- **Griffin Hess** - University of Rochester
- **Chenliang Xu** - University of Rochester
- **Niaz Abdolrahim** - University of Rochester

## 🙏 Acknowledgments

- National Nuclear Security Administration (NNSA) under grant NA0004078
- National Science Foundation (NSF) under grant 2202124
- Department of Energy (DOE) under award DE-SC0020340

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**Project Website**: [https://niaz60.github.io/OpenXRD/](https://niaz60.github.io/OpenXRD/)
