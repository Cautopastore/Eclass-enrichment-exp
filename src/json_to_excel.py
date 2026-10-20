import argparse
import json
import os

import pandas as pd

from config import (
    OUTPUTS_DIR,
    PROVIDERS,
    DEFAULT_PROVIDER,
    PROVIDER_ENV_VAR,
)


parser = argparse.ArgumentParser(
    description="Convert a stage 4 output into an Excel file."
)
parser.add_argument(
    "--provider",
    choices=list(PROVIDERS),
    default=os.getenv(PROVIDER_ENV_VAR, DEFAULT_PROVIDER),
    help="Which provider's output to convert (default: LLM_PROVIDER or deepseek).",
)
args = parser.parse_args()

INPUT_FILE = OUTPUTS_DIR / args.provider / "stage_4_items.jsonl"
OUTPUT_FILE = INPUT_FILE.with_name("eclass_final.xlsx")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Stage 4 output not found for provider '{args.provider}':\n{INPUT_FILE}"
    )



rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        rows.append(json.loads(line))


df = pd.DataFrame(rows)

df.to_excel(
    OUTPUT_FILE,
    index=False,
    engine="openpyxl"
)

print(f"Converted {len(df)} rows.")
print(f"Output: {OUTPUT_FILE}")
