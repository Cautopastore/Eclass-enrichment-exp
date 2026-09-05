
import json
import time
import re

from openai import OpenAI

from config import (
    ITEMS_FILE,
    STAGE_3_OUTPUT,
    STAGE_4_OUTPUT,
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
# Load items.md
# ============================================================

def load_items(filepath):
    """
    Parse items.md into:

    {
        "27180101": {
            "name": "...",
            "definition": "..."
        },
        ...
    }
    """

    text = filepath.read_text(encoding="utf-8")

    items = {}

    # Expected format:
    #
    # ## 27180101 — Item name
    #
    # Definition...
    #

    pattern = re.compile(
        r"^##\s+(\d{8})\s+—\s+(.+?)\s*$",
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):

        item_id = match.group(1)
        item_name = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        body = text[start:end].strip()

        # Remove markdown separators
        body = body.replace("---", "").strip()

        items[item_id] = {
            "name": item_name,
            "definition": body,
        }

    return items


# ============================================================
# Filter items according to subgroup
# ============================================================

def get_items_for_subgroup(all_items, subgroup_id):
    """
    Example:

    subgroup_id = 27180100

    returns only:

    27180101
    27180102
    27180103
    ...

    """

    if not subgroup_id:
        return {}

    subgroup_prefix = str(subgroup_id)[:6]

    return {
        item_id: data
        for item_id, data in all_items.items()
        if item_id.startswith(subgroup_prefix)
    }


# ============================================================
# Build item context
# ============================================================

def build_items_context(items):

    lines = []

    for item_id, data in items.items():

        lines.append(
            f"ID: {item_id}\n"
            f"NAME: {data['name']}\n"
            f"DEFINITION: {data['definition']}\n"
        )

    return "\n---\n".join(lines)


# ============================================================
# Classification
# ============================================================

def classify_item(
    component_description,
    family_id,
    family_name,
    group_id,
    group_name,
    subgroup_id,
    subgroup_name,
    items,
):
    """
    Ask DeepSeek to select the best ECLASS item.
    """

    items_context = build_items_context(items)

    system_prompt = """
You are an expert in the ECLASS classification system.

Your task is to classify a component into exactly ONE ECLASS ITEM.

The component has already been classified into an ECLASS FAMILY,
GROUP, and SUBGROUP.

You will receive:

1. The component description
2. The previously selected family
3. The previously selected group
4. The previously selected subgroup
5. The candidate items belonging ONLY to that subgroup

Select the single item that best matches the component.

IMPORTANT RULES:

- You MUST select exactly one item from the provided candidates.
- Do NOT invent an ECLASS ID.
- The item_id MUST be one of the IDs provided in the candidates.
- The item_name MUST correspond exactly to the selected ID.
- Use both the item name and definition when making the decision.
- The previously selected family, group, and subgroup are authoritative.
- Do not select an item belonging to another subgroup.
- Confidence must be a number between 0 and 1.
- Do not provide an explanation.
- Do not provide Markdown.
- Return ONLY valid JSON.

Required JSON format:

{
    "item_id": "XXXXXXXX",
    "item_name": "Item name",
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

SELECTED SUBGROUP:
ID: {subgroup_id}
NAME: {subgroup_name}

CANDIDATE ITEMS:

{items_context}
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
                "item_id",
                "item_name",
                "confidence",
            ]

            for key in required_keys:

                if key not in result:
                    raise ValueError(
                        f"Missing key: {key}"
                    )

            item_id = str(result["item_id"])

            # ------------------------------------------------
            # Validate ID exists in candidates
            # ------------------------------------------------

            if item_id not in items:

                raise ValueError(
                    f"Model returned invalid item_id: "
                    f"{item_id}"
                )

            # ------------------------------------------------
            # Validate hierarchy
            # ------------------------------------------------

            if not item_id.startswith(
                str(subgroup_id)[:6]
            ):

                raise ValueError(
                    f"Item {item_id} does not belong "
                    f"to subgroup {subgroup_id}"
                )

            # ------------------------------------------------
            # Always use official ECLASS name
            # ------------------------------------------------

            official_name = items[item_id]["name"]

            result["item_id"] = item_id
            result["item_name"] = official_name

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
        "item_id": None,
        "item_name": None,
        "confidence": 0.0,
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load items
    # --------------------------------------------------------

    print("Loading items...")

    all_items = load_items(ITEMS_FILE)

    print(
        f"Loaded {len(all_items)} items."
    )

    # --------------------------------------------------------
    # Load Stage 3 output
    # --------------------------------------------------------

    if not STAGE_3_OUTPUT.exists():

        raise FileNotFoundError(
            f"Stage 3 output not found:\n"
            f"{STAGE_3_OUTPUT}"
        )

    stage_3_rows = []

    with open(
        STAGE_3_OUTPUT,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            stage_3_rows.append(
                json.loads(line)
            )

    print(
        f"Loaded {len(stage_3_rows)} components "
        f"from Stage 3."
    )

    # --------------------------------------------------------
    # Detect already processed rows
    # --------------------------------------------------------

    processed_rows = set()

    if STAGE_4_OUTPUT.exists():

        with open(
            STAGE_4_OUTPUT,
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
        f"Already processed in Stage 4: "
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
        STAGE_4_OUTPUT,
        "a",
        encoding="utf-8"
    )

    try:

        for index, row in enumerate(stage_3_rows):

            source_row = row.get("source_row")

            # -----------------------------------------------
            # Skip already processed
            # -----------------------------------------------

            if source_row in processed_rows:

                skipped += 1
                continue

            print(
                f"\n[{index + 1}/{len(stage_3_rows)}]"
            )

            component_description = (
                row.get("component_description") or ""
            ).strip()

            family_id = row.get("family_id")
            family_name = row.get("family_name")

            group_id = row.get("group_id")
            group_name = row.get("group_name")

            subgroup_id = row.get("subgroup_id")
            subgroup_name = row.get("subgroup_name")

            subgroup_confidence = float(
                row.get("subgroup_confidence") or 0
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
                f"Subgroup: "
                f"{subgroup_id} — {subgroup_name}"
            )

            print(
                f"Subgroup confidence: "
                f"{subgroup_confidence:.3f}"
            )

            # -----------------------------------------------
            # Confidence gate
            # -----------------------------------------------

            if subgroup_confidence < CONFIDENCE_THRESHOLD:

                print(
                    f"  → SKIPPED: subgroup confidence "
                    f"{subgroup_confidence:.3f} < "
                    f"{CONFIDENCE_THRESHOLD:.2f}"
                )

                result = dict(row)

                result["item_id"] = None
                result["item_name"] = None
                result["item_confidence"] = 0.0

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
            # Missing subgroup
            # -----------------------------------------------

            if not subgroup_id:

                print(
                    "  → No subgroup classification. "
                    "Skipping."
                )

                result = dict(row)

                result["item_id"] = None
                result["item_name"] = None
                result["item_confidence"] = 0.0

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
            # Filter items according to subgroup
            # -----------------------------------------------

            candidate_items = get_items_for_subgroup(
                all_items,
                subgroup_id
            )

            print(
                f"Candidate items: "
                f"{len(candidate_items)}"
            )

            if not candidate_items:

                print(
                    f"WARNING: No items found "
                    f"for subgroup {subgroup_id}"
                )

                result = dict(row)

                result["item_id"] = None
                result["item_name"] = None
                result["item_confidence"] = 0.0

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

            classification = classify_item(
                component_description,
                family_id,
                family_name,
                group_id,
                group_name,
                subgroup_id,
                subgroup_name,
                candidate_items,
            )

            # -----------------------------------------------
            # Preserve Stage 3 data
            # -----------------------------------------------

            result = dict(row)

            result["item_id"] = (
                classification["item_id"]
            )

            result["item_name"] = (
                classification["item_name"]
            )

            result["item_confidence"] = (
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

            if classification["item_id"] is None:

                failed += 1

            else:

                print(
                    f"  → Item: "
                    f"{classification['item_id']} "
                    f"— "
                    f"{classification['item_name']}"
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
    print("STAGE 4 COMPLETE")
    print("=" * 60)

    print(
        f"Total Stage 3 rows:          "
        f"{len(stage_3_rows)}"
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
        f"Low-confidence subgroups:    "
        f"{low_confidence}"
    )

    print(
        f"Failed / no classification:  "
        f"{failed}"
    )

    print(
        f"\nOutput: {STAGE_4_OUTPUT}"
    )


if __name__ == "__main__":
    main()
