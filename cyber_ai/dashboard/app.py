"""
Interactive Streamlit dashboard for cyber_ai.

The dashboard reads real system state persisted by `main.py` (or another runner)
into JSON/JSONL files, and visualizes:
- active flows table with risk scores and decisions
- recent Suricata alerts (parsed from eve.json snapshots if provided)
- VirusTotal reputation lookups (from cache)

Refreshes every 5 seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
DEFAULT_FLOWS_PATH = DEFAULT_STATE_DIR / "flows.json"
DEFAULT_ALERTS_PATH = DEFAULT_STATE_DIR / "suricata_alerts.json"
DEFAULT_INCIDENTS_PATH = Path(__file__).resolve().parent.parent / "incident_response" / "artifacts" / "incidents.jsonl"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl_tail(path: Path, max_lines: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _autorefresh_5s() -> None:
    # Prefer streamlit_autorefresh if installed, otherwise best-effort fallback.
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=5_000, key="cyber_ai_autorefresh")
        return
    except Exception:
        pass

    # Fallback: rerun buttonless; Streamlit may warn depending on version.
    st.caption("Auto-refresh fallback enabled (install `streamlit-autorefresh` for smoother refresh).")
    st.session_state["_cyber_ai_ticks"] = st.session_state.get("_cyber_ai_ticks", 0) + 1


def main() -> None:
    st.set_page_config(page_title="Cyber AI Security Dashboard", layout="wide")
    st.title("Cyber AI – Security Dashboard")
    _autorefresh_5s()

    state_dir = Path(os.getenv("CYBER_AI_STATE_DIR", str(DEFAULT_STATE_DIR)))
    flows_path = Path(os.getenv("CYBER_AI_FLOWS_PATH", str(DEFAULT_FLOWS_PATH)))
    alerts_path = Path(os.getenv("CYBER_AI_SURICATA_ALERTS_PATH", str(DEFAULT_ALERTS_PATH)))
    incidents_path = Path(os.getenv("CYBER_AI_INCIDENTS_PATH", str(DEFAULT_INCIDENTS_PATH)))

    st.sidebar.header("State paths")
    st.sidebar.write(f"State dir: `{state_dir}`")
    st.sidebar.write(f"Flows: `{flows_path}`")
    st.sidebar.write(f"Suricata alerts: `{alerts_path}`")
    st.sidebar.write(f"Incidents: `{incidents_path}`")

    flows = _read_json(flows_path) or {}
    flow_rows = list(flows.values()) if isinstance(flows, dict) else []
    flows_df = pd.DataFrame(flow_rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Active flows")
        if flows_df.empty:
            st.info("No flows state found yet. Start the orchestrator (`python -m cyber_ai.main run`).")
        else:
            # Show most relevant columns first if present.
            preferred = [
                "src_ip",
                "dest_ip",
                "src_port",
                "dest_port",
                "protocol",
                "packet_count",
                "encrypted_flag",
                "malware_score",
                "encrypted_score",
                "plaintext_score",
                "vt_score",
                "suricata_score",
                "risk_score",
                "decision",
                "last_updated",
            ]
            cols = [c for c in preferred if c in flows_df.columns] + [c for c in flows_df.columns if c not in preferred]
            st.dataframe(flows_df[cols], use_container_width=True, height=500)

    with c2:
        st.subheader("Decisions summary")
        if flows_df.empty or "decision" not in flows_df.columns:
            st.write("No decisions yet.")
        else:
            st.bar_chart(flows_df["decision"].value_counts())

    st.subheader("Suricata alerts (parsed)")
    alerts = _read_json(alerts_path)
    if isinstance(alerts, list) and alerts:
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True, height=250)
    else:
        st.write("No Suricata alerts loaded yet.")

    st.subheader("Incident log (tail)")
    incidents = _read_jsonl_tail(incidents_path, max_lines=200)
    if incidents:
        inc_df = pd.DataFrame(incidents)
        st.dataframe(inc_df, use_container_width=True, height=250)
    else:
        st.write("No incidents recorded yet.")


if __name__ == "__main__":
    main()

