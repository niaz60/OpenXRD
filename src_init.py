"""
OpenXRD: A Comprehensive Benchmark and Enhancement Framework for LLM/MLLM XRD Question Answering

This package provides tools for evaluating language models on crystallography and X-ray diffraction questions.
"""

__version__ = "1.0.0"
__author__ = "Ali Vosoughi, Ayoub Shahnazari, et al."
__email__ = "support@openxrd.org"

from .evaluation import OpenXRDEvaluator
from .utils import (
    load_dataset,
    load_supporting_materials,
    calculate_metrics,
    format_prompt,
    parse_model_response,
    validate_api_keys,
    NumpyEncoder
)

__all__ = [
    "OpenXRDEvaluator",
    "load_dataset",
    "load_supporting_materials", 
    "calculate_metrics",
    "format_prompt",
    "parse_model_response",
    "validate_api_keys",
    "NumpyEncoder"
]
