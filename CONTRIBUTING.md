# Contributing to OpenXRD

We welcome contributions to the OpenXRD project! This document provides guidelines for contributing to the codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/OpenXRD.git
   cd OpenXRD
   ```
3. Set up the development environment (see below)
4. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git for version control

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues in the codebase
- **Feature additions**: Add new functionality or models
- **Documentation improvements**: Enhance README, docstrings, or guides
- **Performance optimizations**: Improve code efficiency
- **Test coverage**: Add or improve tests
- **Examples and tutorials**: Create usage examples

### Before You Start

1. Check existing issues to see if your contribution is already being worked on
2. For large changes, create an issue first to discuss the approach
3. Make sure your contribution aligns with the project's goals

## Pull Request Process

1. **Update your fork** with the latest changes from the main repository:
   ```bash
   git remote add upstream https://github.com/niaz60/OpenXRD.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch** from main:
   ```bash
   git checkout -b feature/descriptive-name
   ```

3. **Make your changes** following the coding standards below

4. **Test your changes**:
   ```bash
   python -m pytest tests/
   python scripts/evaluate_openai.py --model gpt-4 --mode closedbook  # Example test
   ```

5. **Commit your changes** with descriptive commit messages:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/descriptive-name
   ```

7. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots if applicable
   - Test results

### Pull Request Requirements

- [ ] Code follows the project's coding standards
- [ ] All tests pass
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

## Issue Guidelines

### Reporting Bugs

When reporting bugs, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details**:
  - Python version
  - Operating system
  - Package versions
- **Error messages** and stack traces
- **Minimal reproducible example** if possible

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed implementation** approach (if you have ideas)
- **Alternative solutions** you've considered

### Questions and Support

For questions about usage:

- Check the documentation first
- Search existing issues
- Use clear, specific titles
- Provide context about what you're trying to achieve

## Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **clear docstrings** for all public functions and classes
- Use **meaningful variable and function names**

### Code Organization

- Keep functions focused and small
- Use descriptive module and file names
- Group related functionality together
- Avoid circular imports

### Documentation

- **Docstrings**: Use Google-style docstrings
  ```python
  def evaluate_model(model_name: str, mode: str) -> Dict[str, Any]:
      """
      Evaluate a model on the XRD benchmark.
      
      Args:
          model_name: Name of the model to evaluate
          mode: Evaluation mode ('closedbook' or 'openbook')
          
      Returns:
          Dictionary containing evaluation results
          
      Raises:
          ValueError: If model_name is not supported
      """
  ```

- **Comments**: Use sparingly, prefer self-documenting code
- **README updates**: Update documentation for new features

### Error Handling

- Use appropriate exception types
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

### Example Code Structure

```python
#!/usr/bin/env python3
"""
Module description.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class following project conventions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the class."""
        self.config = config
    
    def process_data(self, data: List[str]) -> Dict[str, Any]:
        """
        Process input data.
        
        Args:
            data: List of input strings
            
        Returns:
            Processing results
        """
        try:
            # Implementation here
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_evaluation.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Write tests for new functionality
- Use descriptive test names
- Test both success and failure cases
- Use appropriate fixtures and mocks

Example test:

```python
import pytest
from src.evaluation import OpenXRDEvaluator


def test_evaluator_initialization():
    """Test that evaluator initializes correctly."""
    evaluator = OpenXRDEvaluator()
    assert len(evaluator.questions) > 0
    

def test_evaluate_model_with_invalid_input():
    """Test error handling for invalid model input."""
    evaluator = OpenXRDEvaluator()
    
    def mock_model_function(prompt: str) -> str:
        raise ValueError("Mock error")
    
    with pytest.raises(ValueError):
        evaluator.evaluate_model(mock_model_function, "test_model")
```

## Model Integration

### Adding New Models

When adding support for new models:

1. **Create a new evaluation script** in `scripts/`
2. **Follow the existing pattern** in `evaluate_openai.py` or `evaluate_gemini.py`
3. **Implement the required interface**:
   - Model loading function
   - Response generation function
   - Error handling
4. **Add configuration** to the model configs
5. **Update documentation** and examples
6. **Add tests** for the new model integration

### Model Evaluation Interface

All model evaluators should implement:

```python
class ModelEvaluator:
    def evaluate_model(self, model_key: str, mode: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a specific model."""
        pass
    
    def evaluate_all_models(self, **kwargs) -> Dict[str, Any]:
        """Evaluate all available models."""
        pass
```

## Documentation

### API Documentation

- Use clear docstrings for all public APIs
- Include examples in docstrings where helpful
- Document parameters, return values, and exceptions

### User Documentation

- Update README.md for new features
- Add examples to demonstrate usage
- Keep installation instructions current

## Getting Help

If you need help with contributing:

- Open an issue with the "question" label
- Join our discussion forum (if available)
- Reach out to the maintainers

## Recognition

Contributors will be acknowledged in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic papers (for research contributions)

Thank you for contributing to OpenXRD! Your efforts help advance the field of AI-assisted crystallography research.
