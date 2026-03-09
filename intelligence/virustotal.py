"""
intelligence/virustotal.py
Real VirusTotal v3 API integration for file hash and URL reputation lookups.
Reads VT_API_KEY from .env file or environment variable. Respects rate limits. Caches results.
"""

import os
import time
import base64
import hashlib
import requests
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any

# ── Load .env file ─────────────────────────────────────────────────────────────
# Search for .env in the cyber_ai directory and project root
try:
    from dotenv import load_dotenv
    _module_dir = Path(__file__).resolve().parent
    # Try cyber_ai/.env first, then project root/.env
    for _env_path in [
        _module_dir.parent / ".env",          # cyber_ai/.env
        _module_dir.parent.parent / ".env",   # project root/.env
    ]:
        if _env_path.exists():
            load_dotenv(_env_path, override=True)
            print(f"[VirusTotal] Loaded API key from {_env_path}")
            break
except ImportError:
    pass  # python-dotenv not installed — fall back to os.environ

# ── Configuration ──────────────────────────────────────────────────────────────
VT_BASE_URL = "https://www.virustotal.com/api/v3"
# Free-tier: 4 requests/minute → minimum 15s between requests
RATE_LIMIT_DELAY = 15.0
_last_request_time = 0.0


def _get_api_key() -> str:
    """Retrieve the VirusTotal API key from the VT_API_KEY environment variable."""
    key = os.environ.get("VT_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "VT_API_KEY environment variable is not set. "
            "Get a free key at https://www.virustotal.com/gui/join-us"
        )
    return key


def _rate_limit():
    """Enforce rate limiting between API requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed
        print(f"[VirusTotal] Rate limiting: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)
    _last_request_time = time.time()


def _make_request(endpoint: str) -> Optional[Dict[str, Any]]:
    """
    Make an authenticated GET request to the VirusTotal v3 API.

    Args:
        endpoint: The API endpoint path (e.g., '/files/{id}').

    Returns:
        Parsed JSON response dict, or None on error.
    """
    api_key = _get_api_key()
    url = f"{VT_BASE_URL}{endpoint}"
    headers = {"x-apikey": api_key}

    _rate_limit()

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"[VirusTotal] Resource not found: {endpoint}")
            return None
        elif response.status_code == 429:
            print("[VirusTotal] Rate limit exceeded. Waiting 60s before retry...")
            time.sleep(60)
            return _make_request(endpoint)  # Single retry
        else:
            print(f"[VirusTotal] HTTP {response.status_code}: {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print(f"[VirusTotal] Request timed out for {endpoint}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[VirusTotal] Request error: {e}")
        return None


def _extract_reputation_score(data: Dict[str, Any]) -> float:
    """
    Extract a reputation score from VirusTotal analysis stats.

    Returns:
        Float between 0.0 (clean) and 1.0 (malicious).
    """
    if not data:
        return 0.0

    attributes = data.get("data", {}).get("attributes", {})
    stats = attributes.get("last_analysis_stats", {})

    malicious = stats.get("malicious", 0)
    suspicious = stats.get("suspicious", 0)
    undetected = stats.get("undetected", 0)
    harmless = stats.get("harmless", 0)

    total = malicious + suspicious + undetected + harmless
    if total == 0:
        return 0.0

    # Weighted: malicious counts full, suspicious counts half
    threat_score = (malicious + 0.5 * suspicious) / total
    return round(min(threat_score, 1.0), 4)


# ── Cached Public API ──────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def query_vt_hash(hash_str: str) -> float:
    """
    Look up a file hash (MD5, SHA1, or SHA256) on VirusTotal.

    Args:
        hash_str: The file hash to look up.

    Returns:
        Reputation score between 0.0 (clean) and 1.0 (malicious).
    """
    hash_str = hash_str.strip().lower()
    print(f"[VirusTotal] Looking up file hash: {hash_str[:16]}...")
    data = _make_request(f"/files/{hash_str}")
    score = _extract_reputation_score(data)
    print(f"[VirusTotal] Hash reputation score: {score}")
    return score


@lru_cache(maxsize=256)
def query_vt_url(url_str: str) -> float:
    """
    Look up a URL on VirusTotal.

    The URL is base64-encoded (without padding) as required by the v3 API.

    Args:
        url_str: The URL to check.

    Returns:
        Reputation score between 0.0 (clean) and 1.0 (malicious).
    """
    url_str = url_str.strip()
    # VT v3 requires base64url encoding of the URL (no padding)
    url_id = base64.urlsafe_b64encode(url_str.encode()).decode().rstrip("=")
    print(f"[VirusTotal] Looking up URL: {url_str[:60]}...")
    data = _make_request(f"/urls/{url_id}")
    score = _extract_reputation_score(data)
    print(f"[VirusTotal] URL reputation score: {score}")
    return score


def query_vt_hash_detailed(hash_str: str) -> Dict[str, Any]:
    """
    Get the full VirusTotal report for a file hash (not just the score).

    Returns:
        Full API response dict, or empty dict on error.
    """
    hash_str = hash_str.strip().lower()
    data = _make_request(f"/files/{hash_str}")
    return data if data else {}


# ── Convenience / Offline Mode ─────────────────────────────────────────────────

def is_api_key_available() -> bool:
    """Check if the VT_API_KEY environment variable is set."""
    return bool(os.environ.get("VT_API_KEY", ""))


def safe_query_hash(hash_str: str) -> float:
    """Query VT hash, returning 0.0 if API key is missing instead of raising."""
    if not is_api_key_available():
        print("[VirusTotal] API key not set — returning 0.0")
        return 0.0
    try:
        return query_vt_hash(hash_str)
    except Exception as e:
        print(f"[VirusTotal] Error querying hash: {e}")
        return 0.0


def safe_query_url(url_str: str) -> float:
    """Query VT URL, returning 0.0 if API key is missing instead of raising."""
    if not is_api_key_available():
        print("[VirusTotal] API key not set — returning 0.0")
        return 0.0
    try:
        return query_vt_url(url_str)
    except Exception as e:
        print(f"[VirusTotal] Error querying URL: {e}")
        return 0.0


if __name__ == "__main__":
    print(f"VT API key available: {is_api_key_available()}")
    if is_api_key_available():
        # Test with a known benign hash (Windows calc.exe SHA256)
        test_hash = "d378bffb70923139d6a4f546864aa61c8..."
        score = query_vt_hash(test_hash)
        print(f"Test hash score: {score}")
