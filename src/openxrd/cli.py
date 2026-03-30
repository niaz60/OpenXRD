"""Console entry points for OpenXRD."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .constants import DATA_POLICY_LINES
from .data import get_data_status
from .evaluator import evaluate
from .providers import PROVIDERS, has_api_key, provider_env_var


def main_check(argv: list[str] | None = None) -> int:
    """Report install, dataset, and provider status."""

    load_dotenv()
    status = get_data_status()

    print(f"OpenXRD {__version__}")
    print(f"Archive: {'found' if status['archive_exists'] else 'missing'} -> {status['archive_path']}")
    print(f"Extracted data directory: {status['data_dir']}")
    print(f"Extracted dataset: {'ready' if status['data_ready'] else 'missing files'}")
    if status["missing_files"]:
        print("Missing extracted files:")
        for name in status["missing_files"]:
            print(f"  - {name}")
        print("Next step: ./scripts/unzip_dataset.sh --acknowledge")

    print("Provider keys:")
    for provider in sorted(PROVIDERS):
        env_var = provider_env_var(provider)
        label = "present" if has_api_key(provider) else "missing"
        print(f"  - {provider}: {env_var} -> {label}")

    print("Dataset policy:")
    for line in DATA_POLICY_LINES:
        print(f"  - {line}")

    return 0


def main_example(argv: list[str] | None = None) -> int:
    """Run a short example evaluation against the benchmark."""

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a short OpenXRD example evaluation.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", required=True, help="Model identifier for the selected provider.")
    parser.add_argument("--mode", choices=["closedbook", "openbook"], default="closedbook")
    parser.add_argument(
        "--material-kind",
        choices=["generated", "expert_reviewed"],
        default="expert_reviewed",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of questions to run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON results.",
    )
    args = parser.parse_args(argv)

    payload = evaluate(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        limit=args.limit,
        material_kind=args.material_kind,
        output_path=args.output,
    )

    print(
        f"{payload['provider']}:{payload['model']} -> "
        f"{payload['correct_answers']}/{payload['total_questions']} "
        f"correct ({payload['accuracy']:.2%})"
    )
    for result in payload["results"]:
        outcome = "correct" if result["is_correct"] else "incorrect"
        print(
            f"Q{result['question_id'] + 1}: {outcome}; "
            f"predicted {result['predicted_option_number']} vs "
            f"gold {result['correct_option_number']}"
        )

    if args.output is not None:
        print(f"Saved results to {args.output}")

    return 0
