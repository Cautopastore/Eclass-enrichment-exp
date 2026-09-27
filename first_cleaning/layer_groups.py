
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("eclass_family_clean.jsonl")

#context files 
FAMILIES_FILE = Path("families.md")
GROUPS_FILE = Path("groups.md")
SUBGROUPS_FILE = Path("subgroups.md")
ITEMS_FILE = Path("items.md")


# ============================================================
# STORAGE
# ============================================================

#lists
families = []
groups = []
subgroups = []
items = []


# ============================================================
# READ JSONL
# ============================================================

with INPUT_FILE.open("r", encoding="utf-8") as infile:

    for line_number, line in enumerate(infile, start=1):  #(index item) pairs on iteration

        line = line.strip() # removes whitespaces

        if not line:
            continue

        try:
            entry = json.loads(line) #converts json to dictionary
        except json.JSONDecodeError:
            print(f"WARNING: Invalid JSON on line {line_number}")
            continue

        eclass_id = str(entry.get("id", "")).strip() #id:value , name:value , definition:value

        if len(eclass_id) != 8 or not eclass_id.isdigit():
            print(f"WARNING: Invalid ECLASS ID on line {line_number}: {eclass_id}")
            continue

        name = (entry.get("name") or "").strip()
        definition = (entry.get("definition") or "").strip()

        # ----------------------------------------------------
        # Determine ECLASS layer
        #
        # 23 00 00 00 -> Family
        # 23 01 00 00 -> Group
        # 23 01 01 00 -> Subgroup
        # 23 01 01 01 -> Item
        # ----------------------------------------------------

        #appends to lists based on the ECLASS ID structure, which indicates the hierarchy level of the entry
        if eclass_id[2:] == "000000":
            families.append((eclass_id, name, definition))

        elif eclass_id[4:] == "0000":
            groups.append((eclass_id, name, definition))

        elif eclass_id[6:] == "00":
            subgroups.append((eclass_id, name, definition))

        else:
            items.append((eclass_id, name, definition))


# ============================================================
# SORT
# ============================================================

families.sort(key=lambda x: x[0])
groups.sort(key=lambda x: x[0])
subgroups.sort(key=lambda x: x[0])
items.sort(key=lambda x: x[0])


# ============================================================
# MARKDOWN WRITER
# ============================================================

#see .md output files to check the structure
def write_markdown(filename, title, entries):

    with filename.open("w", encoding="utf-8") as outfile:

        outfile.write(f"# ECLASS {title}\n\n")
        outfile.write(f"Total entries: {len(entries)}\n\n")
        outfile.write("---\n\n")

        for eclass_id, name, definition in entries:

            outfile.write(f"## {eclass_id} — {name}\n\n")

            if definition:
                outfile.write(f"{definition}\n\n")

            outfile.write("---\n\n")


# ============================================================
# WRITE FILES
# ============================================================

write_markdown(
    FAMILIES_FILE,
    "Families",
    families
)

write_markdown(
    GROUPS_FILE,
    "Groups",
    groups
)

write_markdown(
    SUBGROUPS_FILE,
    "Subgroups",
    subgroups
)

write_markdown(
    ITEMS_FILE,
    "Items",
    items
)


# ============================================================
# SUMMARY
# ============================================================

print("ECLASS splitting completed.\n")

print(f"Families:  {len(families):,} -> {FAMILIES_FILE}")
print(f"Groups:    {len(groups):,} -> {GROUPS_FILE}")
print(f"Subgroups: {len(subgroups):,} -> {SUBGROUPS_FILE}")
print(f"Items:     {len(items):,} -> {ITEMS_FILE}")
