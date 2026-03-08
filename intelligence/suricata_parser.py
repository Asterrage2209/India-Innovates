"""
intelligence/suricata_parser.py
Parse real Suricata eve.json (NDJSON) logs and extract structured alert records.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class AlertRecord:
    """Structured representation of a single Suricata IDS alert."""
    timestamp: str = ""
    src_ip: str = ""
    dest_ip: str = ""
    src_port: int = 0
    dest_port: int = 0
    protocol: str = ""
    event_type: str = ""
    # Alert-specific fields
    alert_msg: str = ""
    category: str = ""
    severity: int = 0
    signature_id: int = 0
    signature_rev: int = 0
    action: str = ""
    # Optional metadata
    app_proto: str = ""
    flow_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_suricata_logs(file_path: str, max_alerts: int = 0) -> List[AlertRecord]:
    """
    Parse a real Suricata eve.json file and extract alert events.

    Args:
        file_path: Absolute path to the eve.json NDJSON log file.
        max_alerts: Maximum number of alerts to parse (0 = unlimited).

    Returns:
        List of AlertRecord dataclass instances.
    """
    alerts: List[AlertRecord] = []
    line_count = 0
    error_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                continue

            # Only process alert events
            if record.get("event_type") != "alert":
                continue

            alert_data = record.get("alert", {})

            alert = AlertRecord(
                timestamp=record.get("timestamp", ""),
                src_ip=record.get("src_ip", ""),
                dest_ip=record.get("dest_ip", ""),
                src_port=record.get("src_port", 0),
                dest_port=record.get("dest_port", 0),
                protocol=record.get("proto", ""),
                event_type=record.get("event_type", ""),
                alert_msg=alert_data.get("signature", ""),
                category=alert_data.get("category", ""),
                severity=alert_data.get("severity", 0),
                signature_id=alert_data.get("signature_id", 0),
                signature_rev=alert_data.get("rev", 0),
                action=alert_data.get("action", ""),
                app_proto=record.get("app_proto", ""),
                flow_id=record.get("flow_id"),
            )
            alerts.append(alert)

            if max_alerts > 0 and len(alerts) >= max_alerts:
                break

    print(f"[SuricataParser] Processed {line_count} lines, "
          f"extracted {len(alerts)} alerts, {error_count} parse errors.")
    return alerts


def get_alert_score(alerts: List[AlertRecord]) -> float:
    """
    Compute a normalized threat score from a list of Suricata alerts.

    The score is based on the severity distribution of the alerts.
    Suricata severity: 1 = high, 2 = medium, 3 = low (inverted scale).

    Returns:
        Float between 0.0 (no threat) and 1.0 (critical threat).
    """
    if not alerts:
        return 0.0

    # Invert Suricata severity: sev 1 → weight 1.0, sev 2 → 0.66, sev 3 → 0.33
    severity_weights = {1: 1.0, 2: 0.66, 3: 0.33}
    total_weight = sum(
        severity_weights.get(a.severity, 0.33) for a in alerts
    )
    # Normalize: cap at 1.0 based on alert count
    max_possible = len(alerts) * 1.0
    score = min(total_weight / max_possible, 1.0) if max_possible > 0 else 0.0
    return round(score, 4)


def get_alerts_for_ip(alerts: List[AlertRecord], ip: str) -> List[AlertRecord]:
    """Filter alerts involving a specific source or destination IP."""
    return [a for a in alerts if a.src_ip == ip or a.dest_ip == ip]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"d:\Programs\Projects\india innovates\suricata-logs\eve.json"
    alerts = parse_suricata_logs(path, max_alerts=1000)
    print(f"\nFirst 3 alerts:")
    for a in alerts[:3]:
        print(f"  [{a.severity}] {a.alert_msg} | {a.src_ip}:{a.src_port} -> {a.dest_ip}:{a.dest_port}")
    print(f"\nAlert score (first 1000): {get_alert_score(alerts)}")
