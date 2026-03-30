"""Short OpenXRD example using either OpenAI or OpenRouter.

Before running:
  1. ./scripts/install.sh
  2. source .venv/bin/activate
  3. ./scripts/unzip_dataset.sh --acknowledge
  4. export OPENAI_API_KEY=... or export OPENROUTER_API_KEY=...
  5. export OPENXRD_MODEL=...

OpenRouter can be used with Claude-family models by setting, for example:
  export OPENXRD_PROVIDER=openrouter
  export OPENXRD_MODEL=anthropic/claude-3.5-sonnet
"""

from __future__ import annotations

import os

from openxrd import evaluate


def main() -> None:
    provider = os.getenv("OPENXRD_PROVIDER", "openai")
    model = os.getenv("OPENXRD_MODEL")
    if not model:
        raise SystemExit("Set OPENXRD_MODEL before running this example.")

    results = evaluate(
        provider=provider,
        model=model,
        mode="closedbook",
        limit=3,
    )
    print(
        f"{provider}:{model} -> {results['correct_answers']}/{results['total_questions']} "
        f"correct ({results['accuracy']:.2%})"
    )


if __name__ == "__main__":
    main()
