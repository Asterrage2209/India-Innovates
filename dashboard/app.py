"""
dashboard/app.py — Interactive Streamlit Dashboard for the AI Cyber Security System
Features: Active flows, risk scores, malware results, phishing alerts,
          Suricata alerts, VirusTotal reputations, and incident log.
Auto-refreshes every 5 seconds.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cyber_ai.intelligence.suricata_parser import (
    parse_suricata_logs, get_alert_score, AlertRecord
)
from cyber_ai.intelligence.virustotal import is_api_key_available, safe_query_url
from cyber_ai.firewall.flow_table import FlowTable, get_flow_table
from cyber_ai.risk_engine.risk_scoring import compute_risk_score, classify_risk
from cyber_ai.zero_trust.access_control import ZeroTrustController
from cyber_ai.incident_response.response_engine import get_response_engine

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberAI — Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for dark premium look ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h2 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        color: #a8a8b3;
        font-size: 0.85rem;
        margin: 5px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .risk-critical { color: #ff4757; font-weight: 700; }
    .risk-high { color: #ffa502; font-weight: 600; }
    .risk-medium { color: #ffd32a; }
    .risk-low { color: #2ed573; }

    .status-blocked { color: #ff4757; font-weight: 700; }
    .status-monitor { color: #ffa502; font-weight: 600; }
    .status-allow { color: #2ed573; }

    .header-bar {
        background: linear-gradient(90deg, #0f3460 0%, #533483 50%, #e94560 100%);
        padding: 15px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-bar h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .header-bar p {
        color: rgba(255,255,255,0.8);
        margin: 5px 0 0 0;
        font-size: 0.9rem;
    }

    .alert-card {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
SURICATA_LOG = PROJECT_ROOT / "suricata-logs" / "eve.json"


# ── Data Loading & Caching ─────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_suricata_alerts(max_alerts: int = 2000):
    """Load and cache Suricata alerts."""
    if not SURICATA_LOG.exists():
        return []
    return parse_suricata_logs(str(SURICATA_LOG), max_alerts=max_alerts)


@st.cache_data(ttl=60)
def get_phishing_predictions():
    """Get phishing predictions for test URLs."""
    try:
        from cyber_ai.phishing.url_detector import predict_phishing, load_saved_model
        if not load_saved_model():
            return []
        test_urls = [
            "https://www.google.com",
            "http://192.168.1.1/admin/login.php",
            "https://secure-bank-update.xyz/verify",
            "https://github.com/python/cpython",
            "http://free-prize-winner.com/claim?id=12345",
            "https://paypal-secure-login.suspicious.com/update",
            "https://stackoverflow.com/questions",
            "http://bit.ly/3x5kZm9",
        ]
        results = []
        for url in test_urls:
            score = predict_phishing(url)
            results.append({
                "URL": url,
                "Phishing Score": round(score, 4),
                "Verdict": "⚠️ Phishing" if score > 0.5 else "✅ Legitimate",
            })
        return results
    except Exception as e:
        return [{"URL": "Error", "Phishing Score": 0, "Verdict": str(e)}]


def build_flow_data_from_alerts(alerts):
    """Build flow table data from real Suricata alerts."""
    ft = get_flow_table()

    # If flow table is empty, populate from alerts
    if ft.flow_count == 0 and alerts:
        for alert in alerts[:500]:
            key = ft.add_packet(
                src_ip=alert.src_ip,
                dest_ip=alert.dest_ip,
                src_port=alert.src_port,
                dest_port=alert.dest_port,
                protocol=alert.protocol,
                payload_size=100,
                encrypted=(alert.dest_port in (443, 8443)),
            )
            # Compute suricata score for this flow's IP
            ip_alerts = [a for a in alerts if a.src_ip == alert.src_ip]
            suri_score = get_alert_score(ip_alerts)

            risk = compute_risk_score(
                suricata_score=suri_score,
                encrypted_score=0.3 if alert.dest_port in (443, 8443) else 0.0,
            )
            decision = "Block" if risk > 0.85 else ("Monitor" if risk > 0.7 else "Allow")
            ft.update_flow_scores(
                key,
                suricata_score=suri_score,
                risk_score=risk,
                decision=decision,
            )

    return ft


# ── Dashboard Layout ───────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="header-bar">
        <h1>🛡️ CyberAI — Security Operations Center</h1>
        <p>AI-Powered Threat Detection • Real-time Monitoring • Zero Trust Access Control</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(ft, alerts):
    """Render the top-level KPI metrics."""
    summary = ft.summary()
    cols = st.columns(6)

    metrics = [
        (summary["total_flows"], "Active Flows", "#00d2ff"),
        (summary["blocked"], "Blocked", "#ff4757"),
        (summary["high_risk"], "High Risk", "#ffa502"),
        (len(alerts), "IDS Alerts", "#e94560"),
        (summary["encrypted"], "Encrypted Flows", "#533483"),
        (summary["total_packets"], "Total Packets", "#2ed573"),
    ]

    for col, (value, label, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: {color}">{value:,}</h2>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)


def render_flow_table(ft):
    """Render the active flows table."""
    st.subheader("📊 Active Network Flows")

    flows = ft.get_all_flows()
    if not flows:
        st.info("No active flows. Run main.py or wait for data ingestion.")
        return

    # Sort by risk score descending
    flows.sort(key=lambda f: f.risk_score, reverse=True)

    data = []
    for f in flows[:100]:  # Show top 100
        risk_class = classify_risk(f.risk_score)
        data.append({
            "Source": f"{f.src_ip}:{f.src_port}",
            "Destination": f"{f.dest_ip}:{f.dest_port}",
            "Protocol": f.protocol,
            "Packets": f.packet_count,
            "Encrypted": "🔒" if f.encrypted_flag else "📄",
            "Risk Score": round(f.risk_score, 3),
            "Risk Level": risk_class,
            "Suricata": round(f.suricata_score, 3),
            "Decision": f.decision,
        })

    df = pd.DataFrame(data)

    # Color-code risk levels
    def color_risk(val):
        colors = {
            "Critical": "background-color: rgba(255,71,87,0.3)",
            "High": "background-color: rgba(255,165,2,0.2)",
            "Medium": "background-color: rgba(255,211,42,0.1)",
            "Low": "background-color: rgba(46,213,115,0.1)",
        }
        return colors.get(val, "")

    def color_decision(val):
        colors = {
            "Block": "color: #ff4757; font-weight: bold",
            "Monitor": "color: #ffa502; font-weight: bold",
            "Allow": "color: #2ed573",
        }
        return colors.get(val, "")

    styled_df = df.style.applymap(color_risk, subset=["Risk Level"])
    styled_df = styled_df.applymap(color_decision, subset=["Decision"])

    st.dataframe(df, use_container_width=True, height=400)


def render_risk_distribution(ft):
    """Render risk score distribution chart."""
    st.subheader("📈 Risk Score Distribution")

    flows = ft.get_all_flows()
    if not flows:
        return

    risk_scores = [f.risk_score for f in flows]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=50,
        marker_color="#e94560",
        opacity=0.8,
    ))
    fig.add_vline(x=0.7, line_dash="dash", line_color="#ffa502",
                  annotation_text="High Risk Threshold")
    fig.add_vline(x=0.85, line_dash="dash", line_color="#ff4757",
                  annotation_text="Block Threshold")
    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Flow Count",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_suricata_alerts(alerts):
    """Render Suricata IDS alerts panel."""
    st.subheader("🚨 Suricata IDS Alerts")

    if not alerts:
        st.info("No Suricata alerts loaded.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Category breakdown
        categories = {}
        for a in alerts:
            categories[a.category] = categories.get(a.category, 0) + 1

        cat_df = pd.DataFrame([
            {"Category": k, "Count": v}
            for k, v in sorted(categories.items(), key=lambda x: -x[1])[:10]
        ])

        fig = px.bar(
            cat_df, x="Count", y="Category", orientation="h",
            color="Count",
            color_continuous_scale=["#533483", "#e94560"],
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Severity breakdown
        sev_counts = {1: 0, 2: 0, 3: 0}
        for a in alerts:
            sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1

        sev_df = pd.DataFrame([
            {"Severity": f"Sev {k} ({'High' if k==1 else 'Med' if k==2 else 'Low'})",
             "Count": v, "Level": k}
            for k, v in sorted(sev_counts.items())
        ])

        fig = px.pie(
            sev_df, values="Count", names="Severity",
            color="Severity",
            color_discrete_map={
                "Sev 1 (High)": "#ff4757",
                "Sev 2 (Med)": "#ffa502",
                "Sev 3 (Low)": "#2ed573",
            },
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent alerts table
    st.markdown("**Recent Alerts**")
    recent = alerts[:50]
    alert_data = [{
        "Time": a.timestamp[:19],
        "Severity": a.severity,
        "Source": f"{a.src_ip}:{a.src_port}",
        "Dest": f"{a.dest_ip}:{a.dest_port}",
        "Alert": a.alert_msg[:80],
        "Category": a.category,
        "Action": a.action,
    } for a in recent]
    st.dataframe(pd.DataFrame(alert_data), use_container_width=True, height=300)


def render_phishing_panel():
    """Render phishing detection results."""
    st.subheader("🎣 Phishing URL Detection")

    results = get_phishing_predictions()
    if not results:
        st.info("Phishing model not trained yet. Run main.py first.")
        return

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Score bar chart
    fig = px.bar(
        df, x="URL", y="Phishing Score",
        color="Phishing Score",
        color_continuous_scale=["#2ed573", "#ffa502", "#ff4757"],
        range_color=[0, 1],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                  annotation_text="Threshold")
    fig.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive checker
    st.markdown("**🔍 Check a URL**")
    user_url = st.text_input("Enter a URL to check:", placeholder="https://example.com")
    if user_url:
        try:
            from cyber_ai.phishing.url_detector import predict_phishing
            score = predict_phishing(user_url)
            if score > 0.5:
                st.error(f"⚠️ **PHISHING DETECTED** — Score: {score:.4f}")
            else:
                st.success(f"✅ **Looks safe** — Score: {score:.4f}")
        except Exception as e:
            st.warning(f"Could not check URL: {e}")


def render_vt_panel():
    """Render VirusTotal integration status."""
    st.subheader("🔎 VirusTotal Integration")

    if is_api_key_available():
        st.success("✅ VT API key is configured")

        vt_url = st.text_input("Check URL on VirusTotal:", placeholder="https://suspicious-site.com")
        if vt_url:
            with st.spinner("Querying VirusTotal..."):
                score = safe_query_url(vt_url)
                if score > 0.5:
                    st.error(f"⚠️ Malicious — VT Score: {score:.4f}")
                elif score > 0.2:
                    st.warning(f"⚡ Suspicious — VT Score: {score:.4f}")
                else:
                    st.success(f"✅ Clean — VT Score: {score:.4f}")
    else:
        st.warning(
            "VirusTotal API key not set. Set `VT_API_KEY` environment "
            "variable to enable real-time lookups."
        )
        st.code("set VT_API_KEY=your_api_key_here", language="shell")


def render_malware_panel():
    """Render malware detection results."""
    st.subheader("🦠 Malware Detection")

    try:
        from cyber_ai.malware_detection.malware_model import get_model_info, load_saved_model
        if not load_saved_model():
            st.info("Malware model not trained yet. Run main.py to train.")
            return

        info = get_model_info()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", info["status"].title())
        with col2:
            st.metric("Classes", info.get("n_classes", "N/A"))
        with col3:
            st.metric("Estimators", info.get("n_estimators", "N/A"))

        st.markdown(f"**Class names**: {', '.join(str(c) for c in info.get('class_names', []))}")

    except Exception as e:
        st.warning(f"Malware module: {e}")


def render_blocked_flows(ft):
    """Render blocked flows and incident log."""
    st.subheader("🚫 Blocked Flows & Incidents")

    blocked = ft.get_blocked_flows()
    if not blocked:
        st.success("No blocked flows currently.")
        return

    data = [{
        "Source": f"{f.src_ip}:{f.src_port}",
        "Destination": f"{f.dest_ip}:{f.dest_port}",
        "Protocol": f.protocol,
        "Risk Score": round(f.risk_score, 3),
        "Malware": round(f.malware_score, 3),
        "Encrypted": round(f.encrypted_score, 3),
        "Suricata": round(f.suricata_score, 3),
    } for f in blocked[:50]]

    st.dataframe(pd.DataFrame(data), use_container_width=True)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    render_header()

    # Load data
    alerts = load_suricata_alerts(max_alerts=2000)
    ft = build_flow_data_from_alerts(alerts)

    # Top metrics
    render_metrics(ft, alerts)

    st.divider()

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Active Flows",
        "🚨 Suricata Alerts",
        "🎣 Phishing Detection",
        "🦠 Malware Detection",
        "🔎 VirusTotal",
        "🚫 Blocked / Incidents",
    ])

    with tab1:
        render_flow_table(ft)
        render_risk_distribution(ft)

    with tab2:
        render_suricata_alerts(alerts)

    with tab3:
        render_phishing_panel()

    with tab4:
        render_malware_panel()

    with tab5:
        render_vt_panel()

    with tab6:
        render_blocked_flows(ft)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        st.markdown(f"**Time**: {time.strftime('%H:%M:%S')}")
        st.markdown(f"**Flows**: {ft.flow_count}")
        st.markdown(f"**Alerts**: {len(alerts)}")
        st.markdown(f"**VT API**: {'✅' if is_api_key_available() else '❌'}")

        st.divider()
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable (5s)", value=True)
        if auto_refresh:
            time.sleep(5)
            st.rerun()

        st.divider()
        st.markdown("### 📊 Flow Summary")
        summary = ft.summary()
        for key, val in summary.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {val}")


if __name__ == "__main__":
    main()
