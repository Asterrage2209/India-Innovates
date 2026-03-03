"""
VirusTotal v3 API integration with caching and rate limiting.

This is a real external integration. Provide your API key via:
- VT_API_KEY environment variable

Implements:
- query_vt_hash(hash_str: str) -> reputation_score: float
- query_vt_url(url_str: str) -> reputation_score: float

Reputation score is derived from VT's analysis stats (real output):
score = malicious / (malicious + harmless) when both are available and >0.

If VT returns only undetected/timeout counts, the score may be 0.0 and the
raw stats will still be cached. No arbitrary scoring is injected.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


VT_BASE_URL = "https://www.virustotal.com/api/v3"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "vt_cache.sqlite"


@dataclass
class VirusTotalConfig:
    api_key: Optional[str] = None
    cache_path: Path = DEFAULT_CACHE_PATH
    # Minimum seconds between requests (public VT keys can be very limited).
    min_interval_seconds: float = 16.0
    timeout_seconds: float = 30.0


class _RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_ts
            sleep_for = self.min_interval_seconds - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_ts = time.time()


_RATE_LIMITERS: Dict[float, _RateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _get_rate_limiter(min_interval_seconds: float) -> _RateLimiter:
    key = float(min_interval_seconds)
    with _RATE_LIMITERS_LOCK:
        rl = _RATE_LIMITERS.get(key)
        if rl is None:
            rl = _RateLimiter(key)
            _RATE_LIMITERS[key] = rl
        return rl


def _get_api_key(cfg: VirusTotalConfig) -> str:
    key = cfg.api_key or os.getenv("VT_API_KEY")
    if not key:
        raise RuntimeError("VT_API_KEY is not set. Provide a real VirusTotal API key via environment variable.")
    return key


def _init_cache(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vt_cache (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                fetched_at INTEGER NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY(kind, key)
            )
            """
        )
        conn.commit()


def _cache_get(db_path: Path, kind: str, key: str) -> Optional[Dict[str, Any]]:
    _init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT response_json FROM vt_cache WHERE kind=? AND key=?",
            (kind, key),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])


def _cache_put(db_path: Path, kind: str, key: str, payload: Dict[str, Any]) -> None:
    _init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO vt_cache(kind, key, fetched_at, response_json) VALUES (?,?,?,?)",
            (kind, key, int(time.time()), json.dumps(payload)),
        )
        conn.commit()


def _analysis_stats_to_score(stats: Dict[str, Any]) -> float:
    """
    Convert VT last_analysis_stats into a reputation score in [0,1].
    Uses only real outputs; no fabricated probabilities.
    """
    malicious = float(stats.get("malicious", 0) or 0)
    harmless = float(stats.get("harmless", 0) or 0)

    denom = malicious + harmless
    if denom <= 0:
        return 0.0
    return float(malicious / denom)


def _vt_get_json(cfg: VirusTotalConfig, endpoint: str) -> Dict[str, Any]:
    api_key = _get_api_key(cfg)
    headers = {"x-apikey": api_key}
    url = f"{VT_BASE_URL}{endpoint}"

    _get_rate_limiter(cfg.min_interval_seconds).wait()

    resp = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)
    if resp.status_code == 429:
        # Respect rate limits deterministically.
        retry_after = float(resp.headers.get("Retry-After", cfg.min_interval_seconds))
        time.sleep(max(retry_after, cfg.min_interval_seconds))
        resp = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)

    resp.raise_for_status()
    return resp.json()


def query_vt_hash(hash_str: str, cfg: Optional[VirusTotalConfig] = None, use_cache: bool = True) -> float:
    """
    Query VT /files/{id} and derive a reputation score from last_analysis_stats.
    """
    cfg = cfg or VirusTotalConfig()
    key = hash_str.strip().lower()
    if use_cache:
        cached = _cache_get(cfg.cache_path, "file", key)
        if cached is not None:
            stats = (
                cached.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
            )
            return _analysis_stats_to_score(stats)

    payload = _vt_get_json(cfg, f"/files/{key}")
    _cache_put(cfg.cache_path, "file", key, payload)

    stats = payload.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    return _analysis_stats_to_score(stats)


def _vt_url_id(url_str: str) -> str:
    """
    VT v3 expects the URL identifier as urlsafe base64 of the URL (no padding).
    """
    u = url_str.strip()
    b64 = base64.urlsafe_b64encode(u.encode("utf-8")).decode("ascii").strip("=")
    return b64


def query_vt_url(url_str: str, cfg: Optional[VirusTotalConfig] = None, use_cache: bool = True) -> float:
    """
    Query VT /urls/{id} and derive a reputation score from last_analysis_stats.
    """
    cfg = cfg or VirusTotalConfig()
    u = url_str.strip()
    url_id = _vt_url_id(u)
    cache_key = hashlib.sha256(u.encode("utf-8")).hexdigest()

    if use_cache:
        cached = _cache_get(cfg.cache_path, "url", cache_key)
        if cached is not None:
            stats = (
                cached.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
            )
            return _analysis_stats_to_score(stats)

    payload = _vt_get_json(cfg, f"/urls/{url_id}")
    _cache_put(cfg.cache_path, "url", cache_key, payload)

    stats = payload.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    return _analysis_stats_to_score(stats)


def get_cached_vt_raw(kind: str, key: str, cfg: Optional[VirusTotalConfig] = None) -> Optional[Dict[str, Any]]:
    """
    Helper to retrieve cached raw VT JSON response for dashboard/forensics.
    """
    cfg = cfg or VirusTotalConfig()
    return _cache_get(cfg.cache_path, kind, key)

