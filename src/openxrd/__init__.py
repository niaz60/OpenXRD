"""Public package interface for OpenXRD."""

from .data import load_questions, load_supporting_materials
from .evaluator import evaluate

__all__ = ["evaluate", "load_questions", "load_supporting_materials"]
__version__ = "1.0.0"
