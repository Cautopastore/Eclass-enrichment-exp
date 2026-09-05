import json
import time
import re
import pandas as pd
from openai import OpenAI

from config import (
    GROUPS_FILE,
    STAGE_1_OUTPUT,
    STAGE_2_OUTPUT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
)


# ============================================================
# DeepSeek client
# ============================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# ============================================================
# Load groups.md
# ============================================================

def load_groups(filepath):
    """
    Parse groups.md into a dictionary:

    {
        "23010000": {
            "name": "...",
            "definition": "..."
        },
        ...
    }
    """

    text = filepath.read_text(encoding="utf-8")

    groups = {}

    # Matches:
    # ## 23010000 — Some group name
    pattern = re.compile(
        r"^##\s+(\d{8})\s+—\s+(.+?)\s*$",
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        group_id = match.group(1)
        group_name = match.group(2).strip()

        # Definition/body extends until next group
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        body = text[start:end].strip()

        # Remove separator if present
        body = body.replace("---", "").strip()

        groups[group_id] = {
            "name": group_name,
            "definition": body,
        }

    return groups


# ============================================================
# Filter groups according to family
# ============================================================

def get_groups_for_family(all_groups, family_id):
    """
    Example:

    family_id = 27000000

    returns only:

    27010000
    27020000
    27030000
    ...

    """

    if not family_id:
        return {}

    family_prefix = str(family_id)[:2]

    return {
        group_id: data
        for group_id, data in all_groups.items()
        if group_id.startswith(family_prefix)
    }


# ============================================================
# Build prompt context
# ============================================================

def build_groups_context(groups):
    """
    Convert the filtered groups into a compact text representation.
    """

    lines = []

    for group_id, data in groups.items():
        lines.append(
            f"ID: {group_id}\n"
            f"NAME: {data['name']}\n"
            f"DEFINITION: {data['definition']}\n"
        )

    return "\n---\n".join(lines)


# ============================================================
# Classification
# ============================================================

def classify_group(component_description, family_id, family_name, groups):
    """
    Ask DeepSeek to select the best ECLASS group.
    """

    groups_context = build_groups_context(groups)

    system_prompt = """
You are an expert in the ECLASS classification system.

Your task is to classify a component into exactly ONE ECLASS GROUP.

The component has already been classified into an ECLASS FAMILY.

You will receive:
1. The component description
2. The previously selected family
3. The candidate groups belonging ONLY to that family

Select the single group that best matches the component.

IMPORTANT RULES:

- You MUST select exactly one group from the provided candidates.
- Do NOT invent an ECLASS ID.
- The group_id MUST be one of the IDs provided in the candidates.
- The group_name MUST correspond exactly to the selected ID.
- Use both the group name and definition when making the decision.
- The previously selected family is authoritative for this step.
- Do not select a group belonging to another family.
- Confidence must be a number between 0 and 1.
- Do not provide an explanation.
- Do not provide Markdown.
- Return ONLY valid JSON.

Required JSON format:

{
    "group_id": "XXXXXXXX",
    "group_name": "Group name",
    "confidence": 0.0
}
""".strip()

    user_prompt = f"""
COMPONENT:
{component_description}

SELECTED FAMILY:
ID: {family_id}
NAME: {family_name}

CANDIDATE GROUPS:

{groups_context}
""".strip()

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
            )

            content = response.choices[0].message.content

            result = json.loads(content)

            # ------------------------------------------------
            # Validate required fields
            # ------------------------------------------------

            required_keys = [
                "group_id",
                "group_name",
                "confidence",
            ]

            for key in required_keys:
                if key not in result:
                    raise ValueError(
                        f"Missing key: {key}"
                    )

            group_id = str(result["group_id"])

            # ------------------------------------------------
            # Validate that the ID actually exists
            # ------------------------------------------------

            if group_id not in groups:
                raise ValueError(
                    f"Model returned invalid group_id: {group_id}"
                )

            # ------------------------------------------------
            # Validate family hierarchy
            # ------------------------------------------------

            if not group_id.startswith(str(family_id)[:2]):
                raise ValueError(
                    f"Group {group_id} does not belong "
                    f"to family {family_id}"
                )

            # ------------------------------------------------
            # Force the official group name
            #
            # This prevents the model from slightly changing
            # the name while keeping the correct ID.
            # ------------------------------------------------

            official_name = groups[group_id]["name"]

            result["group_id"] = group_id
            result["group_name"] = official_name

            # ------------------------------------------------
            # Validate confidence
            # ------------------------------------------------

            confidence = float(result["confidence"])

            if not 0 <= confidence <= 1:
                raise ValueError(
                    f"Invalid confidence: {confidence}"
                )

            result["confidence"] = confidence

            return result

        except Exception as e:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "group_id": None,
        "group_name": None,
        "confidence": 0.0,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("Loading groups...")
    all_groups = load_groups(GROUPS_FILE)

    print(f"Loaded {len(all_groups)} groups.")

    # --------------------------------------------------------
    # Load Stage 1 output
    # --------------------------------------------------------

    if not STAGE_1_OUTPUT.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found:\n{STAGE_1_OUTPUT}"
        )

    stage_1_rows = []

    with open(
        STAGE_1_OUTPUT,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            stage_1_rows.append(json.loads(line))

    print(
        f"Loaded {len(stage_1_rows)} components "
        f"from Stage 1."
    )

    # --------------------------------------------------------
    # Detect already processed rows
    # --------------------------------------------------------

    processed_rows = set()

    if STAGE_2_OUTPUT.exists():

        with open(
            STAGE_2_OUTPUT,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    result = json.loads(line)

                    if "source_row" in result:
                        processed_rows.add(
                            result["source_row"]
                        )

                except json.JSONDecodeError:
                    continue

    print(
        f"Already processed in Stage 2: "
        f"{len(processed_rows)}"
    )

    # --------------------------------------------------------
    # Process incrementally
    # --------------------------------------------------------

    processed_this_run = 0
    skipped = 0
    failed = 0

    output_file = open(
        STAGE_2_OUTPUT,
        "a",
        encoding="utf-8"
    )

    try:

        for index, row in enumerate(stage_1_rows):

            source_row = row.get("source_row")

            # -----------------------------------------------
            # Skip rows already processed
            # -----------------------------------------------

            if source_row in processed_rows:
                skipped += 1
                continue

            print(
                f"\n[{index + 1}/{len(stage_1_rows)}]"
            )

            component_description = (
                row.get("component_description") or ""
            ).strip()

            family_id = row.get("family_id")
            family_name = row.get("family_name")

            print(
                f"Component: {component_description}"
            )

            print(
                f"Family: {family_id} — {family_name}"
            )

            # -----------------------------------------------
            # If Stage 1 failed, preserve the row and leave
            # group classification empty.
            # -----------------------------------------------

            if not family_id:

                print(
                    "  No family classification. "
                    "Skipping group classification."
                )

                result = dict(row)

                result["group_id"] = None
                result["group_name"] = None
                result["group_confidence"] = 0.0

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_file.flush()

                processed_this_run += 1
                failed += 1

                continue

            # -----------------------------------------------
            # Filter groups using the family ID
            # -----------------------------------------------

            candidate_groups = get_groups_for_family(
                all_groups,
                family_id
            )

            print(
                f"Candidate groups: "
                f"{len(candidate_groups)}"
            )

            if not candidate_groups:

                print(
                    f"WARNING: No groups found for "
                    f"family {family_id}"
                )

                result = dict(row)

                result["group_id"] = None
                result["group_name"] = None
                result["group_confidence"] = 0.0

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_file.flush()

                processed_this_run += 1
                failed += 1

                continue

            # -----------------------------------------------
            # Ask DeepSeek
            # -----------------------------------------------

            classification = classify_group(
                component_description,
                family_id,
                family_name,
                candidate_groups,
            )

            # -----------------------------------------------
            # Preserve everything from Stage 1
            # and append Stage 2 information
            # -----------------------------------------------

            result = dict(row)

            result["group_id"] = (
                classification["group_id"]
            )

            result["group_name"] = (
                classification["group_name"]
            )

            result["group_confidence"] = (
                classification["confidence"]
            )

            # -----------------------------------------------
            # Save immediately
            # -----------------------------------------------

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            output_file.flush()

            processed_this_run += 1

            if classification["group_id"] is None:
                failed += 1

            else:
                print(
                    f"  → Group: "
                    f"{classification['group_id']} "
                    f"— {classification['group_name']}"
                )

                print(
                    f"  → Confidence: "
                    f"{classification['confidence']:.3f}"
                )

    except KeyboardInterrupt:

        print(
            "\n\nInterrupted by user. "
            "All completed rows have been saved."
        )

    finally:

        output_file.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)

    print(
        f"Total Stage 1 rows:       {len(stage_1_rows)}"
    )

    print(
        f"Processed this run:       {processed_this_run}"
    )

    print(
        f"Skipped (already done):   {skipped}"
    )

    print(
        f"Failed / no classification: {failed}"
    )

    print(
        f"\nOutput: {STAGE_2_OUTPUT}"
    )


if __name__ == "__main__":
    main()