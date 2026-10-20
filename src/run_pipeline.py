"""
Pipeline orchestrator.

Lets you choose which stage to run and which LLM provider to use,
then launches the matching stage script (which inherits the choice
via the LLM_PROVIDER environment variable).

Usage:
    python run_pipeline.py                       # fully interactive
    python run_pipeline.py --stage group         # provider chosen interactively
    python run_pipeline.py --stage group --provider requesty_qwen
    python run_pipeline.py --stage all
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import PROVIDERS, DEFAULT_PROVIDER, PROVIDER_ENV_VAR


# Stage name -> script file. Order matters for the "all" option.
STAGES = {
    "family": "stage_1_family.py",
    "group": "stage_2_group.py",
    "subgroup": "stage_3_subgroup.py",
    "item": "stage_4_item.py",
}

# Stage scripts live next to this file, so we always run them from here.
SRC_DIR = Path(__file__).resolve().parent


def choose(prompt, options, default):
    """Show a numbered menu and return the chosen option."""

    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        marker = " (default)" if option == default else ""
        print(f"  {index}) {option}{marker}")

    choice = input(f"Choice [1-{len(options)}]: ").strip()

    if not choice:
        return default

    if choice in options:
        return choice

    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]

    print(f"Invalid choice '{choice}', using default '{default}'.")
    return default


def run_stage(stage, provider):
    """Launch one stage script with the selected provider."""

    script = STAGES[stage]
    print(f"\n=== Running '{stage}' with provider '{provider}' ===\n")

    env = {**os.environ, PROVIDER_ENV_VAR: provider}

    subprocess.run(
        [sys.executable, script],
        cwd=SRC_DIR,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ECLASS enrichment pipeline orchestrator."
    )
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        help="Stage to run (family, group, subgroup, item, all).",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        help="LLM provider to use.",
    )
    args = parser.parse_args()

    stage = args.stage or choose(
        "Which stage do you want to run?",
        [*STAGES, "all"],
        "family",
    )

    provider = args.provider or choose(
        "Which LLM provider do you want to use?",
        list(PROVIDERS),
        DEFAULT_PROVIDER,
    )

    stages = list(STAGES) if stage == "all" else [stage]

    for current in stages:
        run_stage(current, provider)


if __name__ == "__main__":
    main()
