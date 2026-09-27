
import json
import time
import re

from openai import OpenAI

from config import (
    SUBGROUPS_FILE,
    STAGE_2_OUTPUT,
    STAGE_3_OUTPUT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    CONFIDENCE_THRESHOLD
)


# ============================================================
# DeepSeek client
# ============================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# ============================================================
# Load subgroups.md
# ============================================================

def load_subgroups(filepath):
    """
    Parse subgroups.md into:

    {
        "27180100": {
            "name": "...",
            "definition": "..."
        },
        ...
    }
    """

    text = filepath.read_text(encoding="utf-8")

    subgroups = {}

    # Expected format:
    #
    # ## 27180100 — Subgroup name
    #
    # Definition...
    #

    pattern = re.compile(
        r"^##\s+(\d{8})\s+—\s+(.+?)\s*$",
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):

        subgroup_id = match.group(1)
        subgroup_name = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        body = text[start:end].strip()

        # Remove markdown separators
        body = body.replace("---", "").strip()

        subgroups[subgroup_id] = {
            "name": subgroup_name,
            "definition": body,
        }

    return subgroups


# ============================================================
# Filter subgroups according to group
# ============================================================

def get_subgroups_for_group(all_subgroups, group_id):
    """
    Example:

    group_id = 27180000

    returns only:

    27180100
    27180200
    27180300
    ...

    """

    if not group_id:
        return {}

    group_prefix = str(group_id)[:4]

    return {
        subgroup_id: data
        for subgroup_id, data in all_subgroups.items()
        if subgroup_id.startswith(group_prefix)
    }


# ============================================================
# Build subgroup context
# ============================================================

def build_subgroups_context(subgroups):

    lines = []

    for subgroup_id, data in subgroups.items():

        lines.append(
            f"ID: {subgroup_id}\n"
            f"NAME: {data['name']}\n"
            f"DEFINITION: {data['definition']}\n"
        )

    return "\n---\n".join(lines)


# ============================================================
# Classification
# ============================================================

def classify_subgroup(
    component_description,
    family_id,
    family_name,
    group_id,
    group_name,
    subgroups,
):
    """
    Ask DeepSeek to select the best ECLASS subgroup.
    """

    subgroups_context = build_subgroups_context(subgroups)

    system_prompt = """
You are an expert in the ECLASS classification system.

Your task is to classify a component into exactly ONE ECLASS SUBGROUP.

The component has already been classified into an ECLASS FAMILY
and an ECLASS GROUP.

You will receive:

1. The component description
2. The previously selected family
3. The previously selected group
4. The candidate subgroups belonging ONLY to that group

Select the single subgroup that best matches the component.

IMPORTANT RULES:

- You MUST select exactly one subgroup from the provided candidates.
- Do NOT invent an ECLASS ID.
- The subgroup_id MUST be one of the IDs provided in the candidates.
- The subgroup_name MUST correspond exactly to the selected ID.
- Use both the subgroup name and definition when making the decision.
- The previously selected family and group are authoritative for this step.
- Do not select a subgroup belonging to another group.
- Confidence must be a number between 0 and 1.
- Do not provide an explanation.
- Do not provide Markdown.
- Return ONLY valid JSON.

Required JSON format:

{
    "subgroup_id": "XXXXXXXX",
    "subgroup_name": "Subgroup name",
    "confidence": 0.0
}
""".strip()

    user_prompt = f"""
COMPONENT:
{component_description}

SELECTED FAMILY:
ID: {family_id}
NAME: {family_name}

SELECTED GROUP:
ID: {group_id}
NAME: {group_name}

CANDIDATE SUBGROUPS:

{subgroups_context}
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
                "subgroup_id",
                "subgroup_name",
                "confidence",
            ]

            for key in required_keys:

                if key not in result:
                    raise ValueError(
                        f"Missing key: {key}"
                    )

            subgroup_id = str(result["subgroup_id"])

            # ------------------------------------------------
            # Validate ID exists in candidates
            # ------------------------------------------------

            if subgroup_id not in subgroups:

                raise ValueError(
                    f"Model returned invalid subgroup_id: "
                    f"{subgroup_id}"
                )

            # ------------------------------------------------
            # Validate hierarchy
            # ------------------------------------------------

            if not subgroup_id.startswith(
                str(group_id)[:4]
            ):

                raise ValueError(
                    f"Subgroup {subgroup_id} does not belong "
                    f"to group {group_id}"
                )

            # ------------------------------------------------
            # Always use official ECLASS name
            # ------------------------------------------------

            official_name = subgroups[subgroup_id]["name"]

            result["subgroup_id"] = subgroup_id
            result["subgroup_name"] = official_name

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
        "subgroup_id": None,
        "subgroup_name": None,
        "confidence": 0.0,
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load subgroups
    # --------------------------------------------------------

    print("Loading subgroups...")

    all_subgroups = load_subgroups(
        SUBGROUPS_FILE
    )

    print(
        f"Loaded {len(all_subgroups)} subgroups."
    )

    # --------------------------------------------------------
    # Load Stage 2 output
    # --------------------------------------------------------

    if not STAGE_2_OUTPUT.exists():

        raise FileNotFoundError(
            f"Stage 2 output not found:\n"
            f"{STAGE_2_OUTPUT}"
        )

    stage_2_rows = []

    with open(
        STAGE_2_OUTPUT,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            stage_2_rows.append(
                json.loads(line)
            )

    print(
        f"Loaded {len(stage_2_rows)} components "
        f"from Stage 2."
    )

    # --------------------------------------------------------
    # Detect already processed rows
    # --------------------------------------------------------

    processed_rows = set()

    if STAGE_3_OUTPUT.exists():

        with open(
            STAGE_3_OUTPUT,
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
        f"Already processed in Stage 3: "
        f"{len(processed_rows)}"
    )

    # --------------------------------------------------------
    # Process incrementally
    # --------------------------------------------------------

    processed_this_run = 0
    skipped = 0
    failed = 0
    low_confidence = 0

    output_file = open(
        STAGE_3_OUTPUT,
        "a",
        encoding="utf-8"
    )

    try:

        for index, row in enumerate(stage_2_rows):

            source_row = row.get("source_row")

            # -----------------------------------------------
            # Skip already processed
            # -----------------------------------------------

            if source_row in processed_rows:

                skipped += 1
                continue

            print(
                f"\n[{index + 1}/{len(stage_2_rows)}]"
            )

            component_description = (
                row.get("component_description") or ""
            ).strip()

            family_id = row.get("family_id")
            family_name = row.get("family_name")

            group_id = row.get("group_id")
            group_name = row.get("group_name")

            group_confidence = float(
                row.get("group_confidence") or 0
            )

            print(
                f"Component: "
                f"{component_description}"
            )

            print(
                f"Family: "
                f"{family_id} — {family_name}"
            )

            print(
                f"Group: "
                f"{group_id} — {group_name}"
            )

            print(
                f"Group confidence: "
                f"{group_confidence:.3f}"
            )

            # -----------------------------------------------
            # Confidence gate
            # -----------------------------------------------

            if group_confidence < CONFIDENCE_THRESHOLD:

                print(
                    f"  → SKIPPED: group confidence "
                    f"{group_confidence:.3f} < "
                    f"{CONFIDENCE_THRESHOLD:.2f}"
                )

                result = dict(row)

                result["subgroup_id"] = None
                result["subgroup_name"] = None
                result["subgroup_confidence"] = 0.0

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                output_file.flush()

                processed_this_run += 1
                low_confidence += 1

                continue

            # -----------------------------------------------
            # Missing group
            # -----------------------------------------------

            if not group_id:

                print(
                    "  → No group classification. "
                    "Skipping."
                )

                result = dict(row)

                result["subgroup_id"] = None
                result["subgroup_name"] = None
                result["subgroup_confidence"] = 0.0

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
            # Filter subgroups according to group
            # -----------------------------------------------

            candidate_subgroups = (
                get_subgroups_for_group(
                    all_subgroups,
                    group_id
                )
            )

            print(
                f"Candidate subgroups: "
                f"{len(candidate_subgroups)}"
            )

            if not candidate_subgroups:

                print(
                    f"WARNING: No subgroups found "
                    f"for group {group_id}"
                )

                result = dict(row)

                result["subgroup_id"] = None
                result["subgroup_name"] = None
                result["subgroup_confidence"] = 0.0

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

            classification = classify_subgroup(
                component_description,
                family_id,
                family_name,
                group_id,
                group_name,
                candidate_subgroups,
            )

            # -----------------------------------------------
            # Preserve Stage 2 data
            # -----------------------------------------------

            result = dict(row)

            result["subgroup_id"] = (
                classification["subgroup_id"]
            )

            result["subgroup_name"] = (
                classification["subgroup_name"]
            )

            result["subgroup_confidence"] = (
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

            if classification["subgroup_id"] is None:

                failed += 1

            else:

                print(
                    f"  → Subgroup: "
                    f"{classification['subgroup_id']} "
                    f"— "
                    f"{classification['subgroup_name']}"
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
    print("STAGE 3 COMPLETE")
    print("=" * 60)

    print(
        f"Total Stage 2 rows:          "
        f"{len(stage_2_rows)}"
    )

    print(
        f"Processed this run:          "
        f"{processed_this_run}"
    )

    print(
        f"Skipped (already done):      "
        f"{skipped}"
    )

    print(
        f"Low-confidence groups:       "
        f"{low_confidence}"
    )

    print(
        f"Failed / no classification:  "
        f"{failed}"
    )

    print(
        f"\nOutput: {STAGE_3_OUTPUT}"
    )


if __name__ == "__main__":
    main()