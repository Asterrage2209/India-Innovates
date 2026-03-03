"""
cyber_ai main orchestrator.

This module wires together:
- Flow table
- Risk scoring
- Zero trust access control
- Incident response
- Suricata alert parsing (eve.json)

IMPORTANT:
- This runner does NOT fabricate traffic. You must provide real inputs:
  - Suricata `eve.json` for IDS alerts (optional but real)
  - Any flow records or packet capture processing can be integrated later
    (PCAP ingestion is not implemented here to avoid inventing parsing logic
     without your real PCAPs and required feature extraction spec).

State output:
- Writes `cyber_ai/state/flows.json` for the Streamlit dashboard.
- Writes `cyber_ai/state/suricata_alerts.json` for parsed Suricata alerts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from cyber_ai.firewall.flow_table import GLOBAL_FLOW_TABLE, FlowKey
from cyber_ai.incident_response.response_engine import GLOBAL_RESPONSE_ENGINE
from cyber_ai.intelligence.suricata_parser import AlertRecord, parse_suricata_logs, suricata_alert_to_score
from cyber_ai.risk_engine.risk_scoring import ScoreInputs, compute_risk_score, RiskScorer
from cyber_ai.zero_trust.access_control import evaluate_session


STATE_DIR = Path(__file__).resolve().parent / "state"
FLOWS_PATH = STATE_DIR / "flows.json"
SURICATA_ALERTS_PATH = STATE_DIR / "suricata_alerts.json"


def _flowkey_from_alert(a: AlertRecord) -> Optional[FlowKey]:
    if not (a.src_ip and a.dest_ip and a.src_port and a.dest_port and a.proto):
        return None
    return (a.src_ip, a.dest_ip, int(a.src_port), int(a.dest_port), str(a.proto).upper())


def _persist_alerts(alerts: List[AlertRecord]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    for a in alerts:
        d = asdict(a)
        # raw can be large; keep it but allow dashboard to handle it
        payload.append(d)
    SURICATA_ALERTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_orchestrator(
    *,
    suricata_eve_path: Optional[Path],
    session_id: str = "default",
    loop_sleep_seconds: float = 2.0,
) -> None:
    """
    Main loop:
    - optionally parse Suricata alerts and update per-flow suricata_score
    - every 30s compute risk score and set decisions via zero-trust policy
    - trigger incident response on Blocked/ReAuth decisions
    - persist state for dashboard
    """
    scorer = RiskScorer(interval_seconds=30.0)
    last_suricata_mtime = 0.0

    while True:
        # Suricata parsing (real file if provided)
        if suricata_eve_path is not None and suricata_eve_path.exists():
            mtime = suricata_eve_path.stat().st_mtime
            if mtime != last_suricata_mtime:
                alerts = parse_suricata_logs(str(suricata_eve_path))
                _persist_alerts(alerts)

                # Update per-flow suricata_score based on max severity score seen
                for a in alerts:
                    fk = _flowkey_from_alert(a)
                    if fk is None:
                        continue
                    entry = GLOBAL_FLOW_TABLE.get_or_create_flow(*fk)
                    new_score = max(entry.suricata_score, suricata_alert_to_score(a))
                    GLOBAL_FLOW_TABLE.update_scores(fk, suricata_score=new_score)

                last_suricata_mtime = mtime

        # Risk scoring every 30s
        if scorer.should_run():
            flows = GLOBAL_FLOW_TABLE.all_flows()
            for fk, entry in flows.items():
                scores = ScoreInputs(
                    malware_score=entry.malware_score,
                    encrypted_score=entry.encrypted_score,
                    plaintext_score=entry.plaintext_score,
                    vt_score=entry.vt_score,
                    suricata_score=entry.suricata_score,
                )
                risk = compute_risk_score(scores)
                decision = evaluate_session(session_id=session_id, risk_score=risk)
                GLOBAL_FLOW_TABLE.update_scores(fk, risk_score=risk, decision=decision)

                # Incident response
                GLOBAL_RESPONSE_ENGINE.handle_decision(
                    fk,
                    GLOBAL_FLOW_TABLE.get_flow(fk) or entry,
                    decision=decision,
                    risk_score=risk,
                    details={"session_id": session_id},
                )

            scorer.mark_ran()

        # Persist flows for the dashboard
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_FLOW_TABLE.save_to_json(FLOWS_PATH)

        time.sleep(loop_sleep_seconds)


def _cmd_run(args: argparse.Namespace) -> None:
    eve = Path(args.suricata_eve) if args.suricata_eve else None
    run_orchestrator(
        suricata_eve_path=eve,
        session_id=args.session_id,
        loop_sleep_seconds=float(args.sleep),
    )


def _cmd_dashboard(_: argparse.Namespace) -> None:
    raise SystemExit(
        "Run the Streamlit dashboard with:\n"
        "  streamlit run cyber_ai/dashboard/app.py\n"
        "from the project root."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cyber_ai")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run orchestrator loop (needs real inputs)")
    run.add_argument("--suricata-eve", dest="suricata_eve", default=os.getenv("SURICATA_EVE_JSON", ""), help="Path to Suricata eve.json")
    run.add_argument("--session-id", dest="session_id", default="default")
    run.add_argument("--sleep", dest="sleep", default="2.0")
    run.set_defaults(func=_cmd_run)

    dash = sub.add_parser("dashboard", help="How to run Streamlit dashboard")
    dash.set_defaults(func=_cmd_dashboard)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

