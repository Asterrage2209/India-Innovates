"""
Flow table for tracking per-flow security metrics.

Each flow is keyed by:
- (src_ip, dest_ip, src_port, dest_port, protocol)

For every flow we maintain:
- packet_count
- encrypted_flag
- encrypted_score
- plaintext_score
- vt_score
- suricata_score
- malware_score
- risk_score
- decision
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal, Any
import json
import threading
import time


FlowKey = Tuple[str, str, int, int, str]
Decision = Literal["Allow", "Monitor", "ReAuthentication Required", "Blocked"]


@dataclass
class FlowEntry:
    src_ip: str
    dest_ip: str
    src_port: int
    dest_port: int
    protocol: str
    packet_count: int = 0
    encrypted_flag: bool = False
    encrypted_score: float = 0.0
    plaintext_score: float = 0.0
    vt_score: float = 0.0
    suricata_score: float = 0.0
    malware_score: float = 0.0
    risk_score: float = 0.0
    decision: Decision = "Allow"
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class FlowTable:
    """
    In-memory flow table with simple thread-safe operations.
    """

    def __init__(self) -> None:
        self._flows: Dict[FlowKey, FlowEntry] = {}
        self._lock = threading.Lock()

    def _make_key(
        self,
        src_ip: str,
        dest_ip: str,
        src_port: int,
        dest_port: int,
        protocol: str,
    ) -> FlowKey:
        return (src_ip, dest_ip, int(src_port), int(dest_port), protocol.upper())

    def get_or_create_flow(
        self,
        src_ip: str,
        dest_ip: str,
        src_port: int,
        dest_port: int,
        protocol: str,
    ) -> FlowEntry:
        key = self._make_key(src_ip, dest_ip, src_port, dest_port, protocol)
        with self._lock:
            entry = self._flows.get(key)
            if entry is None:
                entry = FlowEntry(
                    src_ip=src_ip,
                    dest_ip=dest_ip,
                    src_port=int(src_port),
                    dest_port=int(dest_port),
                    protocol=protocol.upper(),
                    last_updated=time.time(),
                )
                self._flows[key] = entry
            return entry

    def add_packet(
        self,
        src_ip: str,
        dest_ip: str,
        src_port: int,
        dest_port: int,
        protocol: str,
        encrypted: Optional[bool] = None,
    ) -> FlowEntry:
        """
        Increment packet count for a flow and optionally set encrypted flag.
        """
        entry = self.get_or_create_flow(src_ip, dest_ip, src_port, dest_port, protocol)
        with self._lock:
            entry.packet_count += 1
            if encrypted is not None:
                entry.encrypted_flag = bool(encrypted)
            entry.last_updated = time.time()
        return entry

    def update_scores(
        self,
        key: FlowKey,
        *,
        encrypted_score: Optional[float] = None,
        plaintext_score: Optional[float] = None,
        vt_score: Optional[float] = None,
        suricata_score: Optional[float] = None,
        malware_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        decision: Optional[Decision] = None,
    ) -> FlowEntry:
        """
        Update per-flow scores and decision.
        """
        with self._lock:
            entry = self._flows.get(key)
            if entry is None:
                raise KeyError(f"Flow {key} not found in table.")

            if encrypted_score is not None:
                entry.encrypted_score = float(encrypted_score)
            if plaintext_score is not None:
                entry.plaintext_score = float(plaintext_score)
            if vt_score is not None:
                entry.vt_score = float(vt_score)
            if suricata_score is not None:
                entry.suricata_score = float(suricata_score)
            if malware_score is not None:
                entry.malware_score = float(malware_score)
            if risk_score is not None:
                entry.risk_score = float(risk_score)
            if decision is not None:
                entry.decision = decision
            entry.last_updated = time.time()

            return entry

    def get_flow(self, key: FlowKey) -> Optional[FlowEntry]:
        with self._lock:
            return self._flows.get(key)

    def all_flows(self) -> Dict[FlowKey, FlowEntry]:
        with self._lock:
            # return a shallow copy to avoid external mutation
            return dict(self._flows)

    def to_serializable(self) -> Dict[str, dict]:
        """
        Represent the flow table as a JSON-serializable dict keyed by a stringified flow key.
        """
        with self._lock:
            return {
                f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}": entry.to_dict()
                for k, entry in self._flows.items()
            }

    def save_to_json(self, path: Path) -> None:
        """
        Persist the current flow table to a JSON file for use by other components
        such as the Streamlit dashboard.
        """
        serializable = self.to_serializable()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        tmp_path.replace(path)


# Simple singleton instance that can be imported across modules.
GLOBAL_FLOW_TABLE = FlowTable()

