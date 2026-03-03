"""
Suricata eve.json parser for alert enrichment.

Suricata's `eve.json` is typically newline-delimited JSON (NDJSON), where each
line is a JSON object representing an event. Alert events have `event_type`
equal to "alert" and contain an `alert` object with fields such as:
- signature / signature_id
- category (classification)
- severity

This module does not fabricate alerts. It only parses real Suricata output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class AlertRecord:
    timestamp: str
    src_ip: Optional[str]
    dest_ip: Optional[str]
    src_port: Optional[int]
    dest_port: Optional[int]
    proto: Optional[str]
    signature: Optional[str]
    signature_id: Optional[int]
    category: Optional[str]
    severity: Optional[int]
    raw: Dict[str, Any]


def _iter_eve_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines; does not create synthetic records.
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def parse_suricata_logs(file_path: str) -> List[AlertRecord]:
    """
    Parse a Suricata `eve.json` file and return a list of alert records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Suricata eve.json file not found at {path}. Provide a real Suricata log file."
        )

    alerts: List[AlertRecord] = []
    for obj in _iter_eve_json_lines(path):
        if obj.get("event_type") != "alert":
            continue

        alert = obj.get("alert", {}) if isinstance(obj.get("alert"), dict) else {}

        rec = AlertRecord(
            timestamp=str(obj.get("timestamp", "")),
            src_ip=obj.get("src_ip"),
            dest_ip=obj.get("dest_ip"),
            src_port=_safe_int(obj.get("src_port")),
            dest_port=_safe_int(obj.get("dest_port")),
            proto=obj.get("proto"),
            signature=alert.get("signature"),
            signature_id=_safe_int(alert.get("signature_id")),
            category=alert.get("category"),
            severity=_safe_int(alert.get("severity")),
            raw=obj,
        )
        alerts.append(rec)
    return alerts


def suricata_alert_to_score(alert: AlertRecord) -> float:
    """
    Convert a Suricata alert to a score in [0,1] based on Suricata severity.

    Suricata severity is typically 1 (high) to 3 (low). This function maps:
    - severity 1 -> 1.0
    - severity 2 -> 0.5
    - severity 3 -> 0.0

    If severity is missing/unknown, returns 0.0.
    """
    if alert.severity is None:
        return 0.0
    if alert.severity <= 1:
        return 1.0
    if alert.severity == 2:
        return 0.5
    return 0.0

