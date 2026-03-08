"""
zero_trust/access_control.py
Zero Trust access control with session-based risk evaluation.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SessionRecord:
    """Tracks a user/device session with its risk history."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_evaluated: float = field(default_factory=time.time)
    current_risk: float = 0.0
    risk_history: List[float] = field(default_factory=list)
    decision: str = "Allowed"
    reauthentication_count: int = 0
    blocked: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Limit risk history in serialization to last 20 entries
        d["risk_history"] = d["risk_history"][-20:]
        return d


class ZeroTrustController:
    """
    Zero Trust access controller.

    Policy thresholds:
        - risk_score > 0.85 → "Blocked"
        - risk_score > 0.70 → "ReAuthentication Required"
        - Otherwise          → "Allowed"
    """

    BLOCK_THRESHOLD = 0.85
    REAUTH_THRESHOLD = 0.70

    def __init__(self):
        self._sessions: Dict[str, SessionRecord] = {}

    def evaluate_session(self, session_id: str, risk_score: float) -> str:
        """
        Evaluate a session against zero trust policy thresholds.

        Args:
            session_id: Unique session identifier.
            risk_score: Current aggregated risk score (0.0 - 1.0).

        Returns:
            Decision string: "Blocked", "ReAuthentication Required", or "Allowed".
        """
        risk_score = max(0.0, min(1.0, risk_score))

        if session_id not in self._sessions:
            self._sessions[session_id] = SessionRecord(session_id=session_id)

        session = self._sessions[session_id]
        session.current_risk = risk_score
        session.risk_history.append(risk_score)
        session.last_evaluated = time.time()

        if risk_score > self.BLOCK_THRESHOLD:
            session.decision = "Blocked"
            session.blocked = True
        elif risk_score > self.REAUTH_THRESHOLD:
            session.decision = "ReAuthentication Required"
            session.reauthentication_count += 1
        else:
            session.decision = "Allowed"
            session.blocked = False

        return session.decision

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session record by ID."""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, SessionRecord]:
        """Return all session records."""
        return dict(self._sessions)

    def get_blocked_sessions(self) -> List[SessionRecord]:
        """Return all currently blocked sessions."""
        return [s for s in self._sessions.values() if s.blocked]

    def revoke_session(self, session_id: str) -> bool:
        """Manually revoke/block a session."""
        if session_id in self._sessions:
            self._sessions[session_id].blocked = True
            self._sessions[session_id].decision = "Blocked"
            return True
        return False

    def clear_session(self, session_id: str) -> bool:
        """Remove a session from tracking."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def summary(self) -> dict:
        """Summary of all sessions."""
        sessions = self._sessions.values()
        return {
            "total_sessions": len(self._sessions),
            "blocked": sum(1 for s in sessions if s.blocked),
            "reauth_required": sum(
                1 for s in sessions if s.decision == "ReAuthentication Required"
            ),
            "allowed": sum(1 for s in sessions if s.decision == "Allowed"),
        }


# ── Module-level convenience function ──────────────────────────────────────────
_controller: Optional[ZeroTrustController] = None


def get_controller() -> ZeroTrustController:
    """Get or create the global ZeroTrustController singleton."""
    global _controller
    if _controller is None:
        _controller = ZeroTrustController()
    return _controller


def evaluate_session(session_id: str, risk_score: float) -> str:
    """Module-level convenience wrapper for ZeroTrustController.evaluate_session."""
    return get_controller().evaluate_session(session_id, risk_score)


if __name__ == "__main__":
    ztc = ZeroTrustController()

    # Test policy thresholds
    assert ztc.evaluate_session("s1", 0.9) == "Blocked"
    assert ztc.evaluate_session("s2", 0.75) == "ReAuthentication Required"
    assert ztc.evaluate_session("s3", 0.5) == "Allowed"
    assert ztc.evaluate_session("s4", 0.3) == "Allowed"
    assert ztc.evaluate_session("s5", 0.86) == "Blocked"
    assert ztc.evaluate_session("s6", 0.71) == "ReAuthentication Required"

    print(f"Session summary: {ztc.summary()}")
    print("All zero trust tests passed!")
