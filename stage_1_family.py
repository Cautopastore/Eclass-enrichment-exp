
import json
import time

import pandas as pd
from openai import OpenAI

from config import (
    COMPONENTS_FILE,
    FAMILIES_FILE,
    STAGE_1_OUTPUT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
)


# ============================================================
# DEEPSEEK CLIENT
# ============================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# ============================================================
# LOAD ECLASS FAMILIES
# ============================================================

print("Loading families.md...")

with FAMILIES_FILE.open("r", encoding="utf-8") as f:
    families_context = f.read()

print(f"Families context loaded: {len(families_context):,} characters")


# ============================================================
# LOAD COMPONENTS
# ============================================================

print("\nLoading components.xlsx...")

df = pd.read_excel(COMPONENTS_FILE)

print(f"Components loaded: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "DESCR_ORD1",
    "DESCR_ORD2",
    "SRTX",
    "PROD",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CREATE COMPONENT DESCRIPTION
# ============================================================

def clean_value(value):
    """Convert Excel values safely to strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_component_description(row):
    """
    Combine the useful component fields.

    UOM is intentionally excluded.
    """

    parts = []

    for column in [
        "DESCR_ORD1",
        "DESCR_ORD2",
        "SRTX",
        "PROD",
    ]:
        value = clean_value(row[column])

        if value:
            parts.append(value)

    return " | ".join(parts)


# ============================================================
# LOAD ALREADY PROCESSED ROWS
# ============================================================

processed_rows = set()

if STAGE_1_OUTPUT.exists():

    print("\nExisting output found.")
    print("Loading previously processed rows...")

    with STAGE_1_OUTPUT.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                result = json.loads(line)

                if "source_row" in result:
                    processed_rows.add(
                        int(result["source_row"])
                    )

            except (json.JSONDecodeError, ValueError):
                print(
                    "WARNING: Invalid line found in existing output. "
                    "Skipping it."
                )

    print(
        f"Already processed: {len(processed_rows):,}"
    )

else:

    print("\nNo existing output found.")
    print("Starting from the beginning.")


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_family(component_description):
    """
    Ask DeepSeek to classify one component into an ECLASS family.
    """

    prompt = f"""
You are an expert in ECLASS classification.

Your task is to classify the following component into the
MOST RELEVANT ECLASS FAMILY using the provided ECLASS family
dictionary.

ECLASS FAMILY DICTIONARY:
-------------------------
{families_context}
-------------------------

COMPONENT:
{component_description}

Select the single most relevant ECLASS family.

Return ONLY valid JSON using exactly this structure:

{{
    "family_id": "XXXXXXXX",
    "family_name": "Family name",
    "confidence": 0.0
}}

Rules:

- family_id must be an ID that actually exists in the provided dictionary.
- family_name must correspond to that ID.
- confidence must be a number between 0 and 1.
- Do not invent an ECLASS ID.
- Do not return explanations.
- Do not return Markdown.
- Return only the JSON object.
"""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise ECLASS classification "
                            "assistant. Always return valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
            )

            content = response.choices[0].message.content

            result = json.loads(content)

            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if "family_id" not in result:
                raise ValueError("Missing family_id")

            if "family_name" not in result:
                raise ValueError("Missing family_name")

            if "confidence" not in result:
                raise ValueError("Missing confidence")

            result["confidence"] = float(
                result["confidence"]
            )

            if not 0 <= result["confidence"] <= 1:
                raise ValueError(
                    f"Invalid confidence: "
                    f"{result['confidence']}"
                )

            return result

        except Exception as e:

            print(
                f"  API error "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "family_id": None,
        "family_name": None,
        "confidence": 0.0,
    }


# ============================================================
# OPEN OUTPUT FILE
# ============================================================

# "a" means append.
# Existing results are preserved.

output_file = STAGE_1_OUTPUT.open(
    "a",
    encoding="utf-8"
)


# ============================================================
# PROCESS COMPONENTS
# ============================================================

total = len(df)

processed_this_run = 0
skipped = 0
failed = 0

print("\nStarting family classification...\n")

try:

    for index, row in df.iterrows():

        # ----------------------------------------------------
        # Skip rows already processed
        # ----------------------------------------------------

        if index in processed_rows:

            skipped += 1
            continue

        source_row = int(index)

        component_description = build_component_description(row)

        print(
            f"[{index + 1}/{total}] "
            f"{component_description[:100]}"
        )

        # ----------------------------------------------------
        # Empty component
        # ----------------------------------------------------

        if not component_description:

            result = {
                "family_id": None,
                "family_name": None,
                "confidence": 0.0,
            }

            failed += 1

        else:

            result = classify_family(
                component_description
            )

            if result["family_id"] is None:
                failed += 1

        # ----------------------------------------------------
        # Build output
        # ----------------------------------------------------

        output = {}

        # Preserve original Excel data
        for column in df.columns:

            value = row[column]

            if pd.isna(value):
                output[column] = None
            else:
                output[column] = value

        # Add source row
        output["source_row"] = source_row

        # Add merged description
        output["component_description"] = (
            component_description
        )

        # Add classification
        output["family_id"] = result["family_id"]

        output["family_name"] = result["family_name"]

        output["family_confidence"] = (
            result["confidence"]
        )

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        output_file.write(
            json.dumps(
                output,
                ensure_ascii=False
            ) + "\n"
        )

        # Force the data to disk immediately
        output_file.flush()

        processed_this_run += 1

        print(
            f"  → {result['family_id']} | "
            f"{result['family_name']} | "
            f"confidence: {result['confidence']:.2f}"
        )

except KeyboardInterrupt:

    print("\n\nInterrupted by user (Ctrl+C).")

finally:

    output_file.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("STAGE 1")
print("========================================")
print(f"Total components:    {total:,}")
print(f"Processed this run:  {processed_this_run:,}")
print(f"Skipped previously:  {skipped:,}")
print(f"Failed:              {failed:,}")
print(f"Output:              {STAGE_1_OUTPUT}")
print("========================================")

if processed_this_run > 0:
    print(
        "\nAll processed results have been saved "
        "to disk."
    )

print(
    "\nYou can safely run the script again. "
    "Previously processed rows will be skipped."
)

