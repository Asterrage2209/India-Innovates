"""
main.py — End-to-end orchestration of the AI-powered Cyber Security System.
Loads/trains all models, parses Suricata logs, demonstrates the full pipeline:
    detection → scoring → access control → incident response
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Module Imports ─────────────────────────────────────────────────────────────
from cyber_ai.intelligence.suricata_parser import (
    parse_suricata_logs, get_alert_score, get_alerts_for_ip
)
from cyber_ai.intelligence.virustotal import (
    safe_query_hash, safe_query_url, is_api_key_available
)
from cyber_ai.firewall.flow_table import FlowTable, get_flow_table
from cyber_ai.risk_engine.risk_scoring import (
    compute_risk_score, classify_risk, RiskEngine
)
from cyber_ai.zero_trust.access_control import (
    ZeroTrustController, evaluate_session
)
from cyber_ai.incident_response.response_engine import (
    IncidentResponseEngine, get_response_engine
)

# ── Configuration ──────────────────────────────────────────────────────────────
SURICATA_LOG = PROJECT_ROOT / "suricata-logs" / "eve.json"
INCIDENTS_DIR = Path(__file__).resolve().parent / "incidents"


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_malware_training():
    """Train or load the malware detection model."""
    print_header("MALWARE DETECTION MODULE")
    from cyber_ai.malware_detection.malware_model import (
        load_saved_model, train_model, get_model_info
    )

    if load_saved_model():
        print(f"Model info: {get_model_info()}")
    else:
        print("No saved model found — training from scratch...")
        print("NOTE: This may take several minutes due to the large dataset (~1.2GB)")
        results = train_model()
        print(f"Training accuracy: {results['accuracy']:.4f}")


def run_phishing_training():
    """Train or load the phishing URL detector."""
    print_header("PHISHING URL DETECTOR")
    from cyber_ai.phishing.url_detector import (
        load_saved_model, train_model, predict_phishing
    )

    if load_saved_model():
        print("Phishing model loaded.")
    else:
        print("Training phishing URL detector...")
        results = train_model(max_samples=50000)
        print(f"Accuracy: {results['accuracy']:.4f}")

    # Test predictions
    test_urls = [
        "https://www.google.com/search?q=python",
        "http://192.168.1.1/admin/login.php?redirect=bank",
        "https://secure-bank-update.xyz/verify-account",
        "https://github.com/python/cpython",
    ]
    print("\nPhishing predictions:")
    for url in test_urls:
        score = predict_phishing(url)
        print(f"  {'⚠️ PHISH' if score > 0.5 else '✅ SAFE '} [{score:.3f}] {url[:60]}")


def run_plaintext_training():
    """Train or load the plaintext classifier."""
    print_header("PLAINTEXT PAYLOAD CLASSIFIER")
    from cyber_ai.firewall.plaintext_classifier import (
        load_saved_model, train_model, predict_plaintext
    )

    if load_saved_model():
        print("Plaintext classifier loaded.")
    else:
        print("Training plaintext classifier...")
        results = train_model(max_samples=50000)
        print(f"Accuracy: {results['accuracy']:.4f}")


def run_encrypted_cnn():
    """Train or load the encrypted traffic CNN."""
    print_header("ENCRYPTED TRAFFIC CNN")
    from cyber_ai.firewall.encrypted_cnn import load_saved_model, train_model

    if load_saved_model():
        print("Encrypted traffic CNN loaded.")
    else:
        try:
            print("Training encrypted traffic CNN...")
            results = train_model()
            print(f"Accuracy: {results['accuracy']:.4f}")
        except FileNotFoundError as e:
            print(f"Dataset not available: {e}")
            print("The module is ready — download the Mendeley dataset to train.")


def run_suricata_analysis():
    """Parse and analyze real Suricata IDS logs."""
    print_header("SURICATA IDS LOG ANALYSIS")

    if not SURICATA_LOG.exists():
        print(f"Suricata log not found at {SURICATA_LOG}")
        return [], 0.0

    alerts = parse_suricata_logs(str(SURICATA_LOG), max_alerts=5000)
    score = get_alert_score(alerts)

    # Top alert categories
    categories = {}
    for a in alerts:
        categories[a.category] = categories.get(a.category, 0) + 1

    print(f"\nAlert score: {score:.4f}")
    print(f"Total alerts parsed: {len(alerts)}")
    print(f"\nTop alert categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  [{count:5d}] {cat}")

    # Unique IPs
    src_ips = set(a.src_ip for a in alerts)
    print(f"\nUnique source IPs: {len(src_ips)}")

    return alerts, score


def run_flow_simulation(alerts):
    """Simulate flow processing using real Suricata alert data."""
    print_header("FLOW TABLE & RISK SCORING SIMULATION")

    ft = get_flow_table()
    ztc = ZeroTrustController()
    ir_engine = get_response_engine(str(INCIDENTS_DIR))

    # Create flows from real Suricata alerts
    print("Creating flows from Suricata alerts...")
    processed = 0
    for alert in alerts[:500]:  # Process first 500 alerts
        key = ft.add_packet(
            src_ip=alert.src_ip,
            dest_ip=alert.dest_ip,
            src_port=alert.src_port,
            dest_port=alert.dest_port,
            protocol=alert.protocol,
            payload_size=100,
            encrypted=(alert.dest_port in (443, 8443)),
        )

        # Compute per-flow Suricata score
        flow_alerts = get_alerts_for_ip(alerts[:500], alert.src_ip)
        suricata_score = get_alert_score(flow_alerts)

        # Get VT score for source IP (will be 0.0 if no API key)
        vt_score = 0.0  # VT needs hashes/URLs, not IPs

        # Compute risk score
        risk = compute_risk_score(
            malware_score=0.0,
            encrypted_score=0.3 if alert.dest_port in (443, 8443) else 0.0,
            plaintext_score=0.0,
            vt_score=vt_score,
            suricata_score=suricata_score,
        )

        # Determine decision
        if risk > 0.85:
            decision = "Block"
        elif risk > 0.7:
            decision = "Monitor"
        else:
            decision = "Allow"

        ft.update_flow_scores(
            key,
            suricata_score=suricata_score,
            vt_score=vt_score,
            risk_score=risk,
            decision=decision,
        )

        # Zero trust evaluation
        session_id = f"session_{alert.src_ip}"
        zt_decision = ztc.evaluate_session(session_id, risk)

        # Incident response for blocked flows
        flow = ft.get_flow(key)
        if flow and flow.decision == "Block":
            ir_engine.handle_incident(flow)

        processed += 1

    # Print summary
    summary = ft.summary()
    print(f"\nFlow Table Summary:")
    print(f"  Total flows:    {summary['total_flows']}")
    print(f"  Blocked:        {summary['blocked']}")
    print(f"  High risk:      {summary['high_risk']}")
    print(f"  Encrypted:      {summary['encrypted']}")
    print(f"  Total packets:  {summary['total_packets']}")

    # Show top 10 highest risk flows
    print(f"\nTop 10 highest-risk flows:")
    all_flows = ft.get_all_flows()
    all_flows.sort(key=lambda f: f.risk_score, reverse=True)
    for flow in all_flows[:10]:
        print(f"  [{flow.decision:8s}] risk={flow.risk_score:.3f} "
              f"suricata={flow.suricata_score:.3f} "
              f"{flow.src_ip}:{flow.src_port} → {flow.dest_ip}:{flow.dest_port} "
              f"({flow.protocol})")

    # Zero trust summary
    zt_summary = ztc.summary()
    print(f"\nZero Trust Summary:")
    print(f"  Total sessions: {zt_summary['total_sessions']}")
    print(f"  Blocked:        {zt_summary['blocked']}")
    print(f"  ReAuth:         {zt_summary['reauth_required']}")
    print(f"  Allowed:        {zt_summary['allowed']}")

    # Incident response summary
    ir_summary = ir_engine.summary()
    print(f"\nIncident Response Summary:")
    print(f"  Total incidents: {ir_summary['total_incidents']}")
    print(f"  Active blocks:   {ir_summary['active_blocks']}")


def run_vt_demo():
    """Demonstrate VirusTotal integration."""
    print_header("VIRUSTOTAL INTEGRATION")
    if is_api_key_available():
        print("VT API key is set. Running live queries...")
        # Example: Query a known benign hash
        score = safe_query_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        print(f"  Empty file SHA256 score: {score}")
    else:
        print("VT_API_KEY not set — skipping live queries.")
        print("Set the environment variable to enable VirusTotal integration.")
        print("  Windows: set VT_API_KEY=your_key_here")
        print("  Linux:   export VT_API_KEY=your_key_here")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print_header("AI-POWERED CYBER SECURITY SYSTEM")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Train/load ML models
    run_malware_training()
    run_phishing_training()
    run_plaintext_training()
    run_encrypted_cnn()

    # Phase 2: Parse Suricata logs
    alerts, suricata_global_score = run_suricata_analysis()

    # Phase 3: VirusTotal
    run_vt_demo()

    # Phase 4: Flow simulation with real data
    if alerts:
        run_flow_simulation(alerts)

    # Done
    print_header("SYSTEM READY")
    print("All modules initialized. Run the Streamlit dashboard:")
    print(f"  streamlit run {Path(__file__).resolve().parent / 'dashboard' / 'app.py'}")


if __name__ == "__main__":
    main()
