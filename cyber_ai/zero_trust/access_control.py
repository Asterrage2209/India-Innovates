"""
Zero Trust session evaluation based on computed risk scores.

Policy (non-negotiable):
- risk_score > 0.7  -> "ReAuthentication Required"
- risk_score > 0.85 -> "Blocked"
Else: "Allow"

This module maintains lightweight in-memory session state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional


Decision = Literal["Allow", "ReAuthentication Required", "Blocked"]


@dataclass
class SessionState:
    session_id: str
    last_risk_score: float = 0.0
    last_decision: Decision = "Allow"
    last_updated: float = 0.0


class ZeroTrustAccessControl:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def evaluate_session(self, session_id: str, risk_score: float) -> Decision:
        rs = float(risk_score)
        if rs > 0.85:
            decision: Decision = "Blocked"
        elif rs > 0.7:
            decision = "ReAuthentication Required"
        else:
            decision = "Allow"

        st = self._sessions.get(session_id)
        if st is None:
            st = SessionState(session_id=session_id)
            self._sessions[session_id] = st

        st.last_risk_score = rs
        st.last_decision = decision
        st.last_updated = time.time()
        return decision

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)


# Convenience singleton
GLOBAL_ZT = ZeroTrustAccessControl()


def evaluate_session(session_id: str, risk_score: float) -> Decision:
    """
    Module-level API as required by the spec.
    """
    return GLOBAL_ZT.evaluate_session(session_id, risk_score)

