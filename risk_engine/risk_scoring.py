"""
risk_engine/risk_scoring.py
Weighted risk score computation from multiple detection engine outputs.
Formula: risk = 0.3*malware + 0.3*encrypted + 0.2*plaintext + 0.1*vt + 0.1*suricata
"""

import time
import threading
from typing import Optional


# ── Risk Score Weights ─────────────────────────────────────────────────────────
WEIGHTS = {
    "malware":   0.3,
    "encrypted": 0.3,
    "plaintext": 0.2,
    "vt":        0.1,
    "suricata":  0.1,
}


def compute_risk_score(
    malware_score: float = 0.0,
    encrypted_score: float = 0.0,
    plaintext_score: float = 0.0,
    vt_score: float = 0.0,
    suricata_score: float = 0.0,
) -> float:
    """
    Compute a composite risk score from individual detection engine outputs.

    All input scores must be in the range [0.0, 1.0].
    Output is clamped to [0.0, 1.0].

    Formula:
        risk = 0.3 * malware + 0.3 * encrypted + 0.2 * plaintext
             + 0.1 * vt + 0.1 * suricata
    """
    raw = (
        WEIGHTS["malware"]   * max(0.0, min(1.0, malware_score)) +
        WEIGHTS["encrypted"] * max(0.0, min(1.0, encrypted_score)) +
        WEIGHTS["plaintext"] * max(0.0, min(1.0, plaintext_score)) +
        WEIGHTS["vt"]        * max(0.0, min(1.0, vt_score)) +
        WEIGHTS["suricata"]  * max(0.0, min(1.0, suricata_score))
    )
    return round(max(0.0, min(1.0, raw)), 4)


def classify_risk(risk_score: float) -> str:
    """
    Classify a risk score into a human-readable risk level.

    Returns one of: 'Critical', 'High', 'Medium', 'Low'.
    """
    if risk_score > 0.85:
        return "Critical"
    elif risk_score > 0.7:
        return "High"
    elif risk_score > 0.4:
        return "Medium"
    else:
        return "Low"


class RiskEngine:
    """
    Periodic risk engine that sweeps all flows in a FlowTable
    and recomputes their risk scores every `interval` seconds.
    """

    def __init__(self, flow_table, interval: float = 30.0):
        """
        Args:
            flow_table: A FlowTable instance to sweep.
            interval: Seconds between sweeps (default 30).
        """
        self.flow_table = flow_table
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def sweep_once(self) -> int:
        """
        Recompute risk scores for all flows in the flow table.
        Returns the number of flows updated.
        """
        flows = self.flow_table.get_all_flows()
        updated = 0
        for flow in flows:
            new_risk = compute_risk_score(
                malware_score=flow.malware_score,
                encrypted_score=flow.encrypted_score,
                plaintext_score=flow.plaintext_score,
                vt_score=flow.vt_score,
                suricata_score=flow.suricata_score,
            )
            key = (flow.src_ip, flow.dest_ip, flow.src_port, flow.dest_port, flow.protocol)
            # Determine decision based on risk
            if new_risk > 0.85:
                decision = "Block"
            elif new_risk > 0.7:
                decision = "Monitor"
            else:
                decision = "Allow"

            self.flow_table.update_flow_scores(
                key, risk_score=new_risk, decision=decision
            )
            updated += 1
        return updated

    def _run_loop(self):
        """Background loop that periodically sweeps flows."""
        while self._running:
            n = self.sweep_once()
            print(f"[RiskEngine] Swept {n} flows at {time.strftime('%H:%M:%S')}")
            time.sleep(self.interval)

    def start(self):
        """Start the periodic risk engine in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[RiskEngine] Started (interval={self.interval}s)")

    def stop(self):
        """Stop the periodic risk engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[RiskEngine] Stopped")


if __name__ == "__main__":
    # Standalone test
    s = compute_risk_score(0.5, 0.5, 0.5, 0.5, 0.5)
    print(f"Risk score (all 0.5): {s}")  # Should be 0.5
    assert abs(s - 0.5) < 0.001, f"Expected 0.5, got {s}"

    s2 = compute_risk_score(1.0, 1.0, 1.0, 1.0, 1.0)
    print(f"Risk score (all 1.0): {s2}")  # Should be 1.0
    assert abs(s2 - 1.0) < 0.001

    s3 = compute_risk_score(0.0, 0.0, 0.0, 0.0, 0.0)
    print(f"Risk score (all 0.0): {s3}")  # Should be 0.0
    assert abs(s3 - 0.0) < 0.001

    print(f"Risk level for 0.9: {classify_risk(0.9)}")   # Critical
    print(f"Risk level for 0.75: {classify_risk(0.75)}")  # High
    print(f"Risk level for 0.5: {classify_risk(0.5)}")    # Medium
    print(f"Risk level for 0.2: {classify_risk(0.2)}")    # Low
    print("All risk engine tests passed!")
