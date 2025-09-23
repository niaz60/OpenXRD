"""
Setup script for OpenXRD package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version
version = "1.0.0"
if os.path.exists("src/__init__.py"):
    with open("src/__init__.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="openxrd",
    version=version,
    author="Ali Vosoughi, Ayoub Shahnazari, et al.",
    author_email="openxrd@example.com",
    description="A Comprehensive Benchmark and Enhancement Framework for LLM/MLLM XRD Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niaz60/OpenXRD",
    project_urls={
        "Bug Tracker": "https://github.com/niaz60/OpenXRD/issues",
        "Documentation": "https://niaz60.github.io/OpenXRD/",
        "Source Code": "https://github.com/niaz60/OpenXRD",
        "Paper": "https://arxiv.org/abs/2507.09155",
    },
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.15",
        ],
        "visualization": [
            "wordcloud>=1.9.0",
            "seaborn>=0.11.0",
        ],
        "llava": [
            "accelerate>=0.20.0",
            "bitsandbytes>=0.39.0",
            "deepspeed>=0.9.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.15",
            "wordcloud>=1.9.0",
            "seaborn>=0.11.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.39.0",
            "deepspeed>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openxrd-evaluate=scripts.run_all_evaluations:main",
            "openxrd-analyze=scripts.analyze_subtasks:main",
            "openxrd-visualize=scripts.visualize_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "crystallography",
        "x-ray diffraction",
        "machine learning",
        "language models",
        "materials science",
        "benchmark",
        "evaluation",
        "AI",
        "LLM",
        "MLLM",
    ],
)
