import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PROJECT PATHS
# ============================================================

# This file lives in src/, so BASE_DIR is src/ and the repository
# root is one level up.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load secrets from the .env file at the project root.
# Absolute path, so it works no matter which folder you run the script from.
load_dotenv(PROJECT_ROOT / ".env")
    
# ECLASS context files
FAMILIES_FILE = PROJECT_ROOT / "context_files" / "families.md"
GROUPS_FILE = PROJECT_ROOT / "context_files" / "groups.md"
SUBGROUPS_FILE = PROJECT_ROOT / "context_files" / "subgroups.md"
ITEMS_FILE = PROJECT_ROOT / "context_files" / "items.md"

# Input components
COMPONENTS_FILE = PROJECT_ROOT / "input" / "components.xlsx"


# ============================================================
# PROVIDER SELECTION
# ============================================================

# Provider used when a stage is run directly (python stage_1_family.py).
DEFAULT_PROVIDER = "deepseek"

# Environment variable the orchestrator sets to pass the choice down
# to the stage script it launches.
PROVIDER_ENV_VAR = "LLM_PROVIDER"

# Provider for this process. The orchestrator sets LLM_PROVIDER before
# launching a stage; direct runs fall back to DEFAULT_PROVIDER.
PROVIDER = os.getenv(PROVIDER_ENV_VAR, DEFAULT_PROVIDER)


# ============================================================
# STAGE OUTPUT PATHS
# ============================================================

# One folder per provider, so runs of different providers never mix
# and each stage chains within its own provider folder.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROVIDER_OUTPUT_DIR = OUTPUTS_DIR / PROVIDER
PROVIDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_1_OUTPUT = PROVIDER_OUTPUT_DIR / "stage_1_families.jsonl"
STAGE_2_OUTPUT = PROVIDER_OUTPUT_DIR / "stage_2_groups.jsonl"
STAGE_3_OUTPUT = PROVIDER_OUTPUT_DIR / "stage_3_subgroups.jsonl"
STAGE_4_OUTPUT = PROVIDER_OUTPUT_DIR / "stage_4_items.jsonl"


# ============================================================
# LLM PROVIDERS
# ============================================================

# Registry of the LLM providers that can be selected at runtime.
# Add a new provider here and it automatically shows up in the
# orchestrator (run_pipeline.py) selection menu.
PROVIDERS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "json_mode": True,
    },
    "requesty_llama": {
        "api_key": os.getenv("REQUESTY_LLAMA_API_KEY"),
        "base_url": os.getenv(
            "REQUESTY_LLAMA_BASE_URL", "https://router.requesty.ai/v1"
        ),
        "model": os.getenv("REQUESTY_LLAMA_MODEL"),
        # Requesty's routed models reject OpenAI-style JSON mode.
        "json_mode": False,
    },
    "requesty_qwen": {
        "api_key": os.getenv("REQUESTY_QWEN_API_KEY"),
        "base_url": os.getenv(
            "REQUESTY_QWEN_BASE_URL", "https://router.requesty.ai/v1"
        ),
        "model": os.getenv("REQUESTY_QWEN_MODEL"),
        # Requesty's routed models reject OpenAI-style JSON mode.
        "json_mode": False,
    },
}

# Whether the currently selected provider supports OpenAI-style JSON mode.
JSON_MODE = PROVIDERS.get(PROVIDER, {}).get("json_mode", True)


def get_client(provider_name=None):
    """
    Build an OpenAI client for the requested provider.

    Returns a (client, model) tuple. Falls back to DEFAULT_PROVIDER
    when provider_name is None.
    """

    provider_name = provider_name or DEFAULT_PROVIDER

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {', '.join(PROVIDERS)}"
        )

    cfg = PROVIDERS[provider_name]

    if not cfg["api_key"]:
        raise RuntimeError(
            f"No API key set for provider '{provider_name}'. "
            f"Add the matching entry to your .env file."
        )

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    return client, cfg["model"]


def extract_json(text):
    """
    Pull the first JSON object out of an LLM response.

    Handles markdown code fences and any surrounding/trailing text
    (some routed models wrap the answer in commentary). Raises
    ValueError if no JSON object can be found.
    """

    if text is None:
        raise ValueError("Empty response content")

    cleaned = text.strip()

    # Strip ```json ... ``` fences if present.
    cleaned = re.sub(
        r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Try each '{' as the start of a JSON object and keep the first
    # one that decodes completely (ignores trailing extra text).
    decoder = json.JSONDecoder()

    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(
        f"No JSON object found in response: {text[:200]!r}"
    )


def chat_json(client, model, messages, **kwargs):
    """
    Call chat completions and return the parsed JSON result.

    Uses JSON mode only when the selected provider supports it;
    otherwise the JSON is extracted from the raw text.
    """

    params = {"model": model, "messages": messages, **kwargs}

    if JSON_MODE:
        params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**params)

    return extract_json(response.choices[0].message.content)


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

