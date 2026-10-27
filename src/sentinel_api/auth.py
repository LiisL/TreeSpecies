"""Copernicus Data Space / Sentinel Hub authentication helpers.

`fetch_access_token` unifies the client-credentials token POST duplicated in
scripts/download_pixel_stack.py's `get_token()` and (there, only used to
validate credentials up front, not to capture the token for reuse)
scripts/download_sentinel2_daily_statistical.py's `check_token()` / scripts/download_sentinel2_10day_statistical.py's `test_token()`.

`get_config` is canonicalized from scripts/download_sentinel2_daily_statistical.py (byte-identical to
scripts/download_sentinel2_10day_statistical.py's version).
"""

import os
from pathlib import Path

import requests
from sentinelhub import SHConfig


def load_env_file(env_path=None):
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Minimal stand-in for python-dotenv (not a project dependency yet).
    Existing environment variables take precedence over the file.
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def fetch_access_token(client_id, client_secret, token_url, timeout=60):
    r = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        },
        timeout=timeout
    )

    print("TOKEN STATUS:", r.status_code)

    if r.status_code != 200:
        print(r.text)
        raise RuntimeError("Failed to obtain access token.")

    return r.json()["access_token"]


def get_config(client_id, client_secret, base_url, token_url):
    config = SHConfig(use_defaults=True)
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = base_url
    config.sh_token_url = token_url

    print("BASE URL:", config.sh_base_url)
    print("TOKEN URL:", config.sh_token_url)
    print("CLIENT ID present:", bool(config.sh_client_id))
    print("CLIENT SECRET present:", bool(config.sh_client_secret))
    print("CLIENT ID prefix:", config.sh_client_id[:10] if config.sh_client_id else "MISSING")

    return config


def check_token(config):
    fetch_access_token(config.sh_client_id, config.sh_client_secret, config.sh_token_url, timeout=30)
    print("Token check OK")
