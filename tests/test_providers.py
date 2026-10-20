"""
Smoke test: verify each configured provider works through chat_json().

Run from the repo root:

    py tests/test_providers.py
"""

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config


for provider in list(config.PROVIDERS):
    os.environ[config.PROVIDER_ENV_VAR] = provider
    importlib.reload(config)

    client, model = config.get_client(provider)

    try:
        result = config.chat_json(
            client,
            model,
            [{"role": "user", "content": 'Return ONLY JSON: {"ok": true}'}],
        )
        print(f"{provider:16} JSON_MODE={config.JSON_MODE}  OK  {result}")
    except Exception as exc:
        print(
            f"{provider:16} JSON_MODE={config.JSON_MODE}  "
            f"ERR {type(exc).__name__}: {str(exc)[:300]}"
        )
