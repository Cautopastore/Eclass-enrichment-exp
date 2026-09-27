
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ECLASS context files
FAMILIES_FILE = BASE_DIR / "context_files" / "families.md"
GROUPS_FILE = BASE_DIR / "context_files" / "groups.md"
SUBGROUPS_FILE = BASE_DIR / "context_files" / "subgroups.md"
ITEMS_FILE = BASE_DIR / "context_files" / "items.md"

# Input components
COMPONENTS_FILE = BASE_DIR / "components.xlsx"

# Stage outputs
STAGE_1_OUTPUT = BASE_DIR / "outputs" / "stage_1_families.jsonl"
STAGE_2_OUTPUT = BASE_DIR / "outputs" / "stage_2_groups.jsonl"
STAGE_3_OUTPUT = BASE_DIR / "outputs" / "stage_3_subgroups.jsonl"
STAGE_4_OUTPUT = BASE_DIR / "outputs" / "stage_4_items.jsonl"


# ============================================================
# DEEPSEEK
# ============================================================

DEEPSEEK_API_KEY = "sk-e9983f8733854bf6be1a6dafa56f8b6d"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_MODEL = "deepseek-v4-flash"

# ============================================================
# Confidence threshold filtering between stages
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

#actually implemented in stage_3_subgroup.py

# ============================================================
# LLM PARAMETERS
# ============================================================

# Keep this low because we want deterministic classification.
TEMPERATURE = 0.0

# Maximum number of retries when an API request fails.
MAX_RETRIES = 3

# Seconds between retries.
RETRY_DELAY = 2
