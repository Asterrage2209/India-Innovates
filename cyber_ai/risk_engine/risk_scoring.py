"""
Risk scoring engine.

Every 30 seconds compute a flow risk score as:

risk_score =
0.3 * malware_score +
0.3 * encrypted_score +
0.2 * plaintext_score +
0.1 * vt_score +
0.1 * suricata_score

All component scores are expected to be real outputs from models/integrations
and to be within [0,1]. This module does not fabricate scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreInputs:
    malware_score: float = 0.0
    encrypted_score: float = 0.0
    plaintext_score: float = 0.0
    vt_score: float = 0.0
    suricata_score: float = 0.0


def compute_risk_score(scores: ScoreInputs) -> float:
    rs = (
        0.3 * float(scores.malware_score)
        + 0.3 * float(scores.encrypted_score)
        + 0.2 * float(scores.plaintext_score)
        + 0.1 * float(scores.vt_score)
        + 0.1 * float(scores.suricata_score)
    )
    # Keep within [0,1] without injecting arbitrary logic beyond bounds.
    return max(0.0, min(1.0, float(rs)))


class RiskScorer:
    """
    Simple scheduler-friendly risk scorer.
    """

    def __init__(self, interval_seconds: float = 30.0) -> None:
        self.interval_seconds = float(interval_seconds)
        self._last_run = 0.0

    def should_run(self, now: Optional[float] = None) -> bool:
        t = time.time() if now is None else float(now)
        return (t - self._last_run) >= self.interval_seconds

    def mark_ran(self, now: Optional[float] = None) -> None:
        self._last_run = time.time() if now is None else float(now)

