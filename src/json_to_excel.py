import argparse
import json
from pathlib import Path

import pandas as pd

from config import DEFAULT_PROVIDER, OUTPUTS_DIR, PROVIDERS


def choose_provider():
    """Prompt for the provider whose stage 4 output should be converted."""

    providers = list(PROVIDERS)

    print("\nWhich provider's stage 4 output do you want to convert?")
    for index, provider in enumerate(providers, 1):
        marker = " (default)" if provider == DEFAULT_PROVIDER else ""
        print(f"  {index}) {provider}{marker}")

    while True:
        choice = input(f"Choice [1-{len(providers)}]: ").strip()

        if not choice:
            return DEFAULT_PROVIDER
        if choice in providers:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(providers):
            return providers[int(choice) - 1]

        print(f"Invalid choice '{choice}'. Please choose a listed provider.")


def convert_to_excel(provider, outputs_dir=OUTPUTS_DIR):
    """Convert one provider's stage 4 JSONL output to an Excel workbook."""

    input_file = Path(outputs_dir) / provider / "stage_4_items.jsonl"
    output_file = input_file.with_name("eclass_final.xlsx")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Stage 4 output not found for provider '{provider}':\n{input_file}"
        )

    rows = []
    with input_file.open("r", encoding="utf-8") as input_stream:
        for line in input_stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataframe = pd.DataFrame(rows)
    dataframe.to_excel(output_file, index=False, engine="openpyxl")

    print(f"Converted {len(dataframe)} rows.")
    print(f"Output: {output_file}")
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a provider's stage 4 output into an Excel file."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        help="Provider whose stage 4 output to convert (skips the interactive menu).",
    )
    args = parser.parse_args(argv)

    provider = args.provider if args.provider is not None else choose_provider()
    convert_to_excel(provider)


if __name__ == "__main__":
    main()
