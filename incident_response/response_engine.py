"""
incident_response/response_engine.py
Incident response engine: block rules, forensic snapshots, and event logging.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class IncidentEvent:
    """A logged security incident event."""
    timestamp: str
    event_type: str  # "block", "alert", "forensic_snapshot"
    flow_key: str
    risk_score: float
    decision: str
    details: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class IncidentResponseEngine:
    """
    Handles incident response actions:
    - Adds block rules for high-risk flows
    - Saves forensic snapshots as JSON files
    - Maintains an in-memory incident log
    """

    def __init__(self, incidents_dir: str = "incidents"):
        self.incidents_dir = os.path.abspath(incidents_dir)
        os.makedirs(self.incidents_dir, exist_ok=True)
        self._block_list: List[str] = []  # List of blocked flow keys
        self._incident_log: List[IncidentEvent] = []

    def handle_incident(self, flow_record) -> IncidentEvent:
        """
        Process an incident for a blocked/high-risk flow.

        Args:
            flow_record: A FlowRecord (from flow_table.py) with flow details.

        Returns:
            The logged IncidentEvent.
        """
        flow_key = (
            f"{flow_record.src_ip}:{flow_record.src_port}->"
            f"{flow_record.dest_ip}:{flow_record.dest_port}"
            f"/{flow_record.protocol}"
        )
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 1. Add block rule
        if flow_key not in self._block_list:
            self._block_list.append(flow_key)
            print(f"[IR] Block rule added: {flow_key}")

        # 2. Save forensic snapshot
        snapshot = {
            "timestamp": ts,
            "flow_key": flow_key,
            "flow_details": flow_record.to_dict() if hasattr(flow_record, 'to_dict') else str(flow_record),
            "risk_score": flow_record.risk_score,
            "decision": flow_record.decision,
            "scores": {
                "malware": flow_record.malware_score,
                "encrypted": flow_record.encrypted_score,
                "plaintext": flow_record.plaintext_score,
                "vt": flow_record.vt_score,
                "suricata": flow_record.suricata_score,
            }
        }
        snapshot_filename = f"incident_{ts.replace(':', '-')}_{flow_key.replace('/', '_').replace(':', '-').replace('>', '')}.json"
        snapshot_path = os.path.join(self.incidents_dir, snapshot_filename)
        try:
            with open(snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
            print(f"[IR] Forensic snapshot saved: {snapshot_path}")
        except OSError as e:
            print(f"[IR] Error saving snapshot: {e}")

        # 3. Log event
        event = IncidentEvent(
            timestamp=ts,
            event_type="block",
            flow_key=flow_key,
            risk_score=flow_record.risk_score,
            decision=flow_record.decision,
            details=snapshot.get("scores", {}),
        )
        self._incident_log.append(event)
        print(f"[IR] Incident logged: {event.event_type} for {flow_key} "
              f"(risk={flow_record.risk_score})")

        return event

    def get_block_list(self) -> List[str]:
        """Return the current list of blocked flow keys."""
        return list(self._block_list)

    def get_incident_log(self) -> List[IncidentEvent]:
        """Return the full incident event log."""
        return list(self._incident_log)

    def is_blocked(self, src_ip: str, dest_ip: str,
                   src_port: int, dest_port: int, protocol: str) -> bool:
        """Check if a specific flow is in the block list."""
        flow_key = f"{src_ip}:{src_port}->{dest_ip}:{dest_port}/{protocol}"
        return flow_key in self._block_list

    def summary(self) -> dict:
        """Incident response summary."""
        return {
            "total_incidents": len(self._incident_log),
            "active_blocks": len(self._block_list),
            "block_list": self._block_list[-10:],  # Last 10
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_engine: Optional[IncidentResponseEngine] = None


def get_response_engine(incidents_dir: str = "incidents") -> IncidentResponseEngine:
    """Get or create the global incident response engine singleton."""
    global _engine
    if _engine is None:
        _engine = IncidentResponseEngine(incidents_dir)
    return _engine


if __name__ == "__main__":
    from cyber_ai.firewall.flow_table import FlowRecord

    engine = IncidentResponseEngine(incidents_dir="test_incidents")

    # Simulate a blocked flow
    test_flow = FlowRecord(
        src_ip="192.168.1.100",
        dest_ip="10.0.0.5",
        src_port=54321,
        dest_port=443,
        protocol="TCP",
        packet_count=150,
        risk_score=0.92,
        decision="Block",
        malware_score=0.8,
        encrypted_score=0.9,
        plaintext_score=0.1,
        vt_score=0.7,
        suricata_score=0.5,
    )

    event = engine.handle_incident(test_flow)
    print(f"\nIncident summary: {engine.summary()}")
    print(f"Block list: {engine.get_block_list()}")
