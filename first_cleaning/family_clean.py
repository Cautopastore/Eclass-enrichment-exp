
import json
from pathlib import Path

# This script lives in first_cleaning/ and operates on files in the
# same folder, so paths are anchored to the script location rather
# than the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# Input/output files
INPUT_FILE = SCRIPT_DIR / "eclass_clean.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "eclass_family_clean.jsonl"

# ECLASS families to keep
ALLOWED_FAMILIES = {"23", "27", "35", "37", "43", "51"}

kept = 0
removed = 0

with INPUT_FILE.open("r", encoding="utf-8") as infile, \
     OUTPUT_FILE.open("w", encoding="utf-8") as outfile:

    for line in infile:
        line = line.strip()

        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"WARNING: Invalid JSON skipped: {line[:100]}")
            continue

        eclass_id = str(entry.get("id", ""))

        # The first two digits identify the ECLASS family
        family = eclass_id[:2]

        if family in ALLOWED_FAMILIES:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1
        else:
            removed += 1

print("Filtering completed.")
print(f"Kept:    {kept}")
print(f"Removed: {removed}")
print(f"Output:  {OUTPUT_FILE}")
