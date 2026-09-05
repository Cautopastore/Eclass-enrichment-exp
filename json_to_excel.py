import json
import pandas as pd
from pathlib import Path


INPUT_FILE = Path("outputs/stage_4_items.jsonl")
OUTPUT_FILE = Path("outputs/eclass_final.xlsx")


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
