"""
Incident response orchestration.

When a flow is blocked or high risk, the response engine:
- records a persistent incident log entry
- stores a forensic snapshot of the flow record and associated artifacts
- optionally writes a block rule to a local rules file used by this system

This module does not fabricate data; it persists what the system observed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from cyber_ai.firewall.flow_table import FlowEntry, FlowKey


DEFAULT_IR_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BLOCK_RULES_PATH = DEFAULT_IR_DIR / "block_rules.json"
DEFAULT_INCIDENT_LOG_PATH = DEFAULT_IR_DIR / "incidents.jsonl"


@dataclass
class IncidentRecord:
    timestamp: float
    flow_key: str
    decision: str
    risk_score: float
    details: Dict[str, Any]


def _flow_key_to_str(key: FlowKey) -> str:
    return f"{key[0]}:{key[1]}:{key[2]}:{key[3]}:{key[4]}"


class ResponseEngine:
    def __init__(
        self,
        ir_dir: Path = DEFAULT_IR_DIR,
        block_rules_path: Path = DEFAULT_BLOCK_RULES_PATH,
        incident_log_path: Path = DEFAULT_INCIDENT_LOG_PATH,
    ) -> None:
        self.ir_dir = ir_dir
        self.block_rules_path = block_rules_path
        self.incident_log_path = incident_log_path
        self.ir_dir.mkdir(parents=True, exist_ok=True)

    def add_block_rule(self, flow_key: FlowKey) -> None:
        """
        Add a block rule for this system's internal enforcement.
        """
        self.ir_dir.mkdir(parents=True, exist_ok=True)
        rules = []
        if self.block_rules_path.exists():
            try:
                rules = json.loads(self.block_rules_path.read_text(encoding="utf-8"))
            except Exception:
                rules = []

        key_str = _flow_key_to_str(flow_key)
        if key_str not in rules:
            rules.append(key_str)
            self.block_rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    def save_forensic_snapshot(self, flow_key: FlowKey, flow_entry: FlowEntry, extra: Optional[Dict[str, Any]] = None) -> Path:
        """
        Persist a snapshot of the flow entry and any extra artifacts.
        """
        ts = int(time.time())
        snap_dir = self.ir_dir / "snapshots" / f"{ts}_{_flow_key_to_str(flow_key).replace(':', '_')}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        payload = {"flow_key": _flow_key_to_str(flow_key), "flow_entry": flow_entry.to_dict(), "extra": extra or {}}
        out_path = snap_dir / "snapshot.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def log_incident(self, incident: IncidentRecord) -> None:
        self.ir_dir.mkdir(parents=True, exist_ok=True)
        with self.incident_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(incident)) + "\n")

    def handle_decision(
        self,
        flow_key: FlowKey,
        flow_entry: FlowEntry,
        *,
        decision: str,
        risk_score: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Main orchestration entry point. Call this when a flow gets a decision.
        """
        if decision not in {"Blocked", "ReAuthentication Required"}:
            return

        if decision == "Blocked":
            self.add_block_rule(flow_key)

        snapshot_path = self.save_forensic_snapshot(flow_key, flow_entry, extra=details)

        self.log_incident(
            IncidentRecord(
                timestamp=time.time(),
                flow_key=_flow_key_to_str(flow_key),
                decision=decision,
                risk_score=float(risk_score),
                details={"snapshot_path": str(snapshot_path), **(details or {})},
            )
        )


GLOBAL_RESPONSE_ENGINE = ResponseEngine()

