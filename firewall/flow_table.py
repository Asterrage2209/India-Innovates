"""
firewall/flow_table.py
Flow table management for the AI-powered firewall.
Maintains per-flow state keyed by 5-tuple, stores all component scores and decisions.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

# Flow key: (src_ip, dest_ip, src_port, dest_port, protocol)
FlowKey = Tuple[str, str, int, int, str]


@dataclass
class FlowRecord:
    """Complete state record for a single network flow."""
    src_ip: str = ""
    dest_ip: str = ""
    src_port: int = 0
    dest_port: int = 0
    protocol: str = ""
    # Counters
    packet_count: int = 0
    byte_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Encryption detection
    encrypted_flag: bool = False
    # Component scores (all 0.0 - 1.0)
    encrypted_score: float = 0.0
    plaintext_score: float = 0.0
    vt_score: float = 0.0
    suricata_score: float = 0.0
    malware_score: float = 0.0
    # Aggregate
    risk_score: float = 0.0
    # Decision: "Allow", "Monitor", "Block", "ReAuthentication Required"
    decision: str = "Allow"

    def to_dict(self) -> dict:
        return asdict(self)


class FlowTable:
    """
    In-memory flow table for tracking active network flows.

    Keyed by 5-tuple: (src_ip, dest_ip, src_port, dest_port, protocol).
    """

    def __init__(self):
        self._flows: Dict[FlowKey, FlowRecord] = {}

    def _make_key(self, src_ip: str, dest_ip: str,
                  src_port: int, dest_port: int, protocol: str) -> FlowKey:
        return (src_ip, dest_ip, src_port, dest_port, protocol.upper())

    def add_packet(self, src_ip: str, dest_ip: str,
                   src_port: int, dest_port: int, protocol: str,
                   payload_size: int = 0, encrypted: bool = False) -> FlowKey:
        """
        Register a packet in the flow table. Creates a new flow if it doesn't exist.

        Returns:
            The FlowKey for this flow.
        """
        key = self._make_key(src_ip, dest_ip, src_port, dest_port, protocol)
        now = time.time()

        if key not in self._flows:
            self._flows[key] = FlowRecord(
                src_ip=src_ip,
                dest_ip=dest_ip,
                src_port=src_port,
                dest_port=dest_port,
                protocol=protocol.upper(),
                packet_count=1,
                byte_count=payload_size,
                first_seen=now,
                last_seen=now,
                encrypted_flag=encrypted,
            )
        else:
            flow = self._flows[key]
            flow.packet_count += 1
            flow.byte_count += payload_size
            flow.last_seen = now
            if encrypted:
                flow.encrypted_flag = True

        return key

    def update_flow_scores(self, key: FlowKey, **scores) -> None:
        """
        Update score components for a flow.

        Accepted keyword args: encrypted_score, plaintext_score,
        vt_score, suricata_score, malware_score, risk_score, decision.
        """
        if key not in self._flows:
            return

        flow = self._flows[key]
        valid_fields = {
            "encrypted_score", "plaintext_score", "vt_score",
            "suricata_score", "malware_score", "risk_score", "decision"
        }
        for field_name, value in scores.items():
            if field_name in valid_fields:
                setattr(flow, field_name, value)

    def get_flow(self, key: FlowKey) -> Optional[FlowRecord]:
        """Get a single flow record by key."""
        return self._flows.get(key)

    def get_all_flows(self) -> List[FlowRecord]:
        """Return all active flows as a list of FlowRecord objects."""
        return list(self._flows.values())

    def get_blocked_flows(self) -> List[FlowRecord]:
        """Return all flows with decision 'Block'."""
        return [f for f in self._flows.values() if f.decision == "Block"]

    def get_high_risk_flows(self, threshold: float = 0.7) -> List[FlowRecord]:
        """Return all flows with risk_score above a threshold."""
        return [f for f in self._flows.values() if f.risk_score >= threshold]

    def remove_flow(self, key: FlowKey) -> bool:
        """Remove a flow from the table."""
        if key in self._flows:
            del self._flows[key]
            return True
        return False

    def clear_expired(self, max_age_seconds: float = 300) -> int:
        """Remove flows older than max_age_seconds. Returns count of removed flows."""
        now = time.time()
        expired_keys = [
            k for k, v in self._flows.items()
            if (now - v.last_seen) > max_age_seconds
        ]
        for k in expired_keys:
            del self._flows[k]
        return len(expired_keys)

    @property
    def flow_count(self) -> int:
        return len(self._flows)

    def summary(self) -> dict:
        """Return a summary of the flow table state."""
        flows = self._flows.values()
        return {
            "total_flows": len(self._flows),
            "blocked": sum(1 for f in flows if f.decision == "Block"),
            "high_risk": sum(1 for f in flows if f.risk_score >= 0.7),
            "encrypted": sum(1 for f in flows if f.encrypted_flag),
            "total_packets": sum(f.packet_count for f in flows),
        }


# ── Module-level singleton for shared state ────────────────────────────────────
_global_flow_table: Optional[FlowTable] = None


def get_flow_table() -> FlowTable:
    """Get or create the global flow table singleton."""
    global _global_flow_table
    if _global_flow_table is None:
        _global_flow_table = FlowTable()
    return _global_flow_table


if __name__ == "__main__":
    ft = FlowTable()
    # Simulate adding packets
    k1 = ft.add_packet("192.168.1.10", "10.0.0.1", 12345, 443, "TCP", 1500, encrypted=True)
    k2 = ft.add_packet("192.168.1.10", "10.0.0.1", 12345, 443, "TCP", 800, encrypted=True)
    k3 = ft.add_packet("192.168.1.20", "8.8.8.8", 54321, 80, "TCP", 500)

    ft.update_flow_scores(k1, encrypted_score=0.8, risk_score=0.75, decision="Monitor")
    ft.update_flow_scores(k3, plaintext_score=0.3, risk_score=0.2, decision="Allow")

    print(f"Flow table summary: {ft.summary()}")
    for flow in ft.get_all_flows():
        print(f"  {flow.src_ip}:{flow.src_port} -> {flow.dest_ip}:{flow.dest_port} "
              f"| pkts={flow.packet_count} risk={flow.risk_score} decision={flow.decision}")
