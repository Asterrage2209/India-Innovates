<h1 align="center">🛡️ CyberAI — AI-Powered Cyber Security System</h1>

<p align="center">
  <b>A comprehensive, end-to-end security operations platform built in Python</b><br>
  Malware Detection • Flow-Aware Firewalling • Phishing Detection • Threat Intelligence • Zero Trust • Incident Response
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&amp;logo=python" alt="Python">
  <img src="https://img.shields.io/badge/ML-scikit--learn-orange?style=flat-square" alt="scikit-learn">
  <img src="https://img.shields.io/badge/DL-PyTorch-red?style=flat-square&amp;logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/dashboard-Streamlit-ff4b4b?style=flat-square&amp;logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/IDS-Suricata-yellow?style=flat-square" alt="Suricata">
  <img src="https://img.shields.io/badge/API-VirusTotal-394eff?style=flat-square" alt="VirusTotal">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Real Data Sources](#real-data-sources)
- [Installation and Setup](#installation-and-setup)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
  - [Malware Detection](#1-malware-detection)
  - [Phishing URL Detector](#2-phishing-url-detector)
  - [Plaintext Classifier](#3-plaintext-classifier)
  - [Encrypted Traffic CNN](#4-encrypted-traffic-cnn)
  - [Flow Table](#5-flow-table)
  - [Risk Engine](#6-risk-engine)
  - [Zero Trust Access Control](#7-zero-trust-access-control)
  - [Incident Response](#8-incident-response)
  - [VirusTotal Integration](#9-virustotal-integration)
  - [Suricata Log Parser](#10-suricata-log-parser)
  - [Orchestrator (main.py)](#11-orchestrator)
  - [Streamlit Dashboard](#12-streamlit-dashboard)
- [Risk Scoring Formula](#risk-scoring-formula)
- [Zero Trust Policy](#zero-trust-policy)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Research References](#research-references)
- [License](#license)

---

## Overview

CyberAI is a **one-stop security operations center (SOC)** that combines machine learning classifiers trained on real datasets with real-time threat intelligence and an interactive monitoring dashboard. Every classifier is trained on genuine labeled data — **no synthetic, fake, or placeholder logic is used anywhere**.

### Key Capabilities

| Capability | Technology | Data Source |
|-----------|-----------|------------|
| **Malware Detection** | RandomForest (scikit-learn) | Real PE/API/DLL feature CSVs (28,000+ samples, 22,000+ features) |
| **Phishing URL Detection** | RandomForest (14 lexical features) | HuggingFace `testpj/phishing-dataset` |
| **Plaintext Payload Analysis** | TF-IDF char n-grams + RandomForest | HuggingFace phishing dataset |
| **Encrypted Traffic Classification** | 1D-CNN (PyTorch) | Mendeley Encrypted Malicious Traffic Dataset |
| **IDS Alert Parsing** | Structured JSON parser | Real Suricata `eve.json` logs (232MB) |
| **Threat Intelligence** | REST API v3 integration | VirusTotal (live lookups) |
| **Risk Scoring** | Weighted multi-factor formula | Real model outputs (no fake scores) |
| **Zero Trust Access** | Session-based threshold policy | Live risk scores |
| **Incident Response** | Automated blocking + forensic snapshots | Flow table decisions |
| **Dashboard** | Streamlit + Plotly | All modules above |

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                  STREAMLIT DASHBOARD                     │
                    │   Flows │ Alerts │ Phishing │ Malware │ VT │ Incidents  │
                    └───────────────────────┬──────────────────────────────────┘
                                            │
                    ┌───────────────────────┴──────────────────────────────────┐
                    │                    ORCHESTRATOR (main.py)                │
                    └───┬──────────┬──────────┬──────────┬──────────┬─────────┘
                        │          │          │          │          │
              ┌─────────┴──┐ ┌────┴────┐ ┌───┴───┐ ┌───┴───┐ ┌───┴────────┐
              │   RISK     │ │  ZERO   │ │INCIDENT│ │ FLOW  │ │ INTELLIGENCE│
              │  ENGINE    │ │ TRUST   │ │RESPONSE│ │ TABLE │ │  VT + IDS  │
              └─────┬──────┘ └────┬────┘ └───┬───┘ └───┬───┘ └───┬────────┘
                    │             │          │         │          │
        ┌───────────┼─────────────┼──────────┼─────────┘          │
        │           │             │          │                    │
  ┌─────┴─────┐ ┌──┴───┐ ┌──────┴────┐ ┌───┴─────┐    ┌────────┴──────┐
  │  MALWARE  │ │PHISH │ │PLAINTEXT  │ │ENCRYPTED│    │  SURICATA     │
  │  MODEL    │ │ URL  │ │CLASSIFIER │ │  CNN    │    │  PARSER       │
  │(RF/PCA)   │ │(RF)  │ │(TF-IDF/RF)│ │(PyTorch)│    │  (eve.json)   │
  └───────────┘ └──────┘ └───────────┘ └─────────┘    └───────────────┘
```

---

## Real Data Sources

| # | Dataset | Format | Size | Location |
|---|---------|--------|------|----------|
| 1 | **API Function Calls** | CSV (28,017 x 21,920) | 1.2 GB | `../dataset/API_Functions.csv` |
| 2 | **DLL Imports** | CSV (28,016 x 631) | 37 MB | `../dataset/DLLs_Imported.csv` |
| 3 | **PE Headers** | CSV (28,014 x 144) | 14 MB | `../dataset/portable_executable.csv` |
| 4 | **Test Set** | CSV (22,690 columns) | 68 MB | `../dataset/test.csv` |
| 5 | **Suricata Logs** | NDJSON (eve.json) | 232 MB | `../suricata-logs/eve.json` |
| 6 | **Phishing URLs** | HuggingFace dataset | Downloaded at runtime | `testpj/phishing-dataset` |
| 7 | **Encrypted Traffic** | CSV (Mendeley) | User-provided | `../data/encrypted_traffic/` |
| 8 | **PCAP Capture** | PCAP | 681 MB | `../2014-04-07_capture-win17.pcap` |

> **Note:** Datasets 1-5 and 8 are already present in the project. Dataset 6 is auto-downloaded via the `datasets` library. Dataset 7 must be manually downloaded from [Mendeley](https://data.mendeley.com/datasets/xw7r4tt54g/1).

---

## Installation and Setup

### Prerequisites

- Python 3.11+
- pip

### Install Dependencies

```bash
cd cyber_ai
pip install -r requirements.txt
```

### Required packages

| Package | Purpose |
|---------|---------|
| `scikit-learn` | ML classifiers, PCA, TF-IDF |
| `pandas`, `numpy` | Data manipulation |
| `torch` | CNN for encrypted traffic |
| `streamlit` | Dashboard |
| `plotly` | Interactive charts |
| `requests` | VirusTotal API |
| `joblib` | Model persistence |
| `datasets` | HuggingFace data loader |
| `matplotlib`, `seaborn` | Static visualizations |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VT_API_KEY` | Optional | VirusTotal API key ([get free key](https://www.virustotal.com/gui/join-us)) |

```bash
# Windows
set VT_API_KEY=your_api_key_here

# Linux/macOS
export VT_API_KEY=your_api_key_here
```

---

## Quick Start

### 1. Train Models and Run Full Pipeline

```bash
python cyber_ai/main.py
```

This will:
- Load and merge the 3 malware CSVs (API + DLL + PE) on SHA256
- Train the RandomForest malware classifier with PCA dimensionality reduction
- Download and train the phishing URL detector from HuggingFace
- Train the plaintext payload classifier
- Attempt to train the encrypted traffic CNN (if dataset is available)
- Parse 5,000 real Suricata IDS alerts
- Simulate flow processing with real risk scoring
- Print evaluation metrics (accuracy, classification report, confusion matrix)

### 2. Launch the Dashboard

```bash
python -m streamlit run cyber_ai/dashboard/app.py
```

Open `http://localhost:8501` in your browser. The dashboard auto-refreshes every 5 seconds.

### 3. Use Individual Modules

```python
# Phishing detection
from cyber_ai.phishing.url_detector import predict_phishing
score = predict_phishing("http://suspicious-login.xyz/verify")
print(f"Phishing probability: {score:.4f}")

# Risk scoring
from cyber_ai.risk_engine.risk_scoring import compute_risk_score
risk = compute_risk_score(
    malware_score=0.8,
    encrypted_score=0.5,
    plaintext_score=0.3,
    vt_score=0.6,
    suricata_score=0.9
)
print(f"Risk score: {risk:.4f}")

# Suricata parsing
from cyber_ai.intelligence.suricata_parser import parse_suricata_logs
alerts = parse_suricata_logs("path/to/eve.json", max_alerts=1000)
```

---

## Module Reference

### 1. Malware Detection

**File:** `malware_detection/malware_model.py`

Trains a **RandomForestClassifier** on merged real malware feature data (API calls + DLL imports + PE headers). The three CSVs are inner-joined on `SHA256`. Dimensionality reduction uses `VarianceThreshold(0.01)` followed by `PCA(200)` to handle 22,000+ features.

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `train_model()` | `(merged_df, n_components, n_estimators) -> dict` | Train and evaluate the classifier |
| `predict_malware()` | `(sample_features: np.ndarray) -> np.ndarray` | Return class probability distribution |
| `predict_malware_label()` | `(sample_features: np.ndarray) -> list` | Return predicted class labels |
| `load_saved_model()` | `() -> bool` | Load from `saved_models/` directory |

**Label Classes:** 7 malware types (0-6)

---

### 2. Phishing URL Detector

**File:** `phishing/url_detector.py`

Extracts **14 lexical features** from URL strings and trains a RandomForest on the HuggingFace `testpj/phishing-dataset`.

**Extracted Features:**

1. URL length
2. Dot count
3. Slash count
4. Hyphen count
5. Underscore count
6. Digit count
7. `@` symbol count
8. `?` symbol count
9. `&` symbol count
10. Shannon entropy
11. Has IP address (binary)
12. TLD length
13. Path length
14. Subdomain count

**Key Function:**

```python
predict_phishing(url_str: str) -> float  # 0.0 = legitimate, 1.0 = phishing
```

---

### 3. Plaintext Classifier

**File:** `firewall/plaintext_classifier.py`

Uses **character-level TF-IDF** (3-5 char n-grams, up to 10,000 features) with a RandomForest to detect malicious plaintext payloads.

**Key Function:**

```python
predict_plaintext(payload_str: str) -> float  # 0.0 = benign, 1.0 = malicious
```

---

### 4. Encrypted Traffic CNN

**File:** `firewall/encrypted_cnn.py`

A **1D-CNN** in PyTorch for classifying encrypted network flows without decryption.

**Architecture:**

```
Conv1d(1, 32, k=3) -> ReLU -> MaxPool(2)
Conv1d(32, 64, k=3) -> ReLU -> MaxPool(2)
Conv1d(64, 128, k=3) -> ReLU -> AdaptiveAvgPool(1)
FC(128, 64) -> ReLU -> Dropout(0.3)
FC(64, num_classes)
```

**Key Function:**

```python
predict_encrypted_traffic(features: np.ndarray) -> float  # 0.0 = benign, 1.0 = malicious
```

**Dataset:** Download from [Mendeley](https://data.mendeley.com/datasets/xw7r4tt54g/1) and extract to `data/encrypted_traffic/`.

---

### 5. Flow Table

**File:** `firewall/flow_table.py`

In-memory flow state management keyed by 5-tuple `(src_ip, dest_ip, src_port, dest_port, protocol)`.

**FlowRecord Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `packet_count` | int | Total packets in this flow |
| `byte_count` | int | Total bytes |
| `encrypted_flag` | bool | Whether flow is encrypted |
| `encrypted_score` | float | CNN classification score |
| `plaintext_score` | float | Plaintext classifier score |
| `vt_score` | float | VirusTotal reputation |
| `suricata_score` | float | IDS alert severity score |
| `malware_score` | float | Malware classification score |
| `risk_score` | float | Aggregate risk (0.0-1.0) |
| `decision` | str | Allow / Monitor / Block |

---

### 6. Risk Engine

**File:** `risk_engine/risk_scoring.py`

Computes a weighted composite risk score from all detection engines.

```
risk_score = 0.3 * malware + 0.3 * encrypted + 0.2 * plaintext
           + 0.1 * vt + 0.1 * suricata
```

Includes a `RiskEngine` class that can run as a background thread, sweeping all flows every 30 seconds to recompute scores.

---

### 7. Zero Trust Access Control

**File:** `zero_trust/access_control.py`

Session-based access control with risk thresholds:

| Risk Score | Decision |
|-----------|----------|
| > 0.85 | **Blocked** |
| > 0.70 | **ReAuthentication Required** |
| <= 0.70 | **Allowed** |

Tracks risk history per session for auditing.

---

### 8. Incident Response

**File:** `incident_response/response_engine.py`

When a flow is blocked or flagged high-risk:

1. **Block rule** is added to an in-memory blocklist
2. **Forensic snapshot** is saved as a JSON file in `incidents/`
3. **Event** is logged with timestamp and all score details

---

### 9. VirusTotal Integration

**File:** `intelligence/virustotal.py`

Real integration with the **VirusTotal v3 API**.

| Function | Endpoint | Description |
|----------|----------|-------------|
| `query_vt_hash(hash)` | `/api/v3/files/{id}` | File hash reputation (MD5/SHA1/SHA256) |
| `query_vt_url(url)` | `/api/v3/urls/{id}` | URL reputation |
| `safe_query_hash(hash)` | N/A | Graceful fallback (returns 0.0 if no API key) |
| `safe_query_url(url)` | N/A | Graceful fallback |

**Features:**
- LRU cache (256 entries) to avoid duplicate requests
- Rate limiting (15s between requests for free-tier compliance)
- Automatic 429 retry with 60s backoff

---

### 10. Suricata Log Parser

**File:** `intelligence/suricata_parser.py`

Parses real **Suricata eve.json** (NDJSON format) and extracts structured `AlertRecord` objects.

**AlertRecord Fields:** `timestamp`, `src_ip`, `dest_ip`, `src_port`, `dest_port`, `protocol`, `alert_msg`, `category`, `severity`, `signature_id`, `action`, `app_proto`, `flow_id`

**Scoring:** Severity-weighted normalization (Sev 1 = 1.0, Sev 2 = 0.66, Sev 3 = 0.33)

---

### 11. Orchestrator

**File:** `main.py`

End-to-end pipeline that:

1. Trains/loads all ML models
2. Parses Suricata logs
3. Demonstrates VirusTotal integration
4. Simulates flow processing from real alert data
5. Applies risk scoring, zero trust evaluation, and incident response
6. Prints comprehensive summary tables

---

### 12. Streamlit Dashboard

**File:** `dashboard/app.py`

Interactive SOC dashboard with dark premium theme and 6 tabs:

| Tab | Content |
|-----|---------|
| Active Flows | Flow table sorted by risk, risk distribution histogram |
| Suricata Alerts | Category bar chart, severity pie chart, alert table |
| Phishing Detection | Model predictions, interactive URL checker |
| Malware Detection | Model status, class information |
| VirusTotal | API status, interactive URL/hash lookup |
| Blocked / Incidents | Blocked flows table, incident log |

**Features:**
- Auto-refresh every 5 seconds
- Real-time sidebar with system status
- Plotly interactive charts with dark theme

---

## Risk Scoring Formula

```
risk_score = 0.3 * malware_score
           + 0.3 * encrypted_score
           + 0.2 * plaintext_score
           + 0.1 * vt_score
           + 0.1 * suricata_score
```

All component scores are clamped to `[0.0, 1.0]`. The output is a single float in `[0.0, 1.0]`.

**Risk Classification:**

| Score Range | Level |
|------------|-------|
| > 0.85 | Critical |
| > 0.70 | High |
| > 0.40 | Medium |
| <= 0.40 | Low |

---

## Zero Trust Policy

```python
if risk_score > 0.85:
    decision = "Blocked"
elif risk_score > 0.70:
    decision = "ReAuthentication Required"
else:
    decision = "Allowed"
```

---

## API Reference

### Core Prediction Functions

```python
# Malware detection
from cyber_ai.malware_detection.malware_model import predict_malware
probabilities = predict_malware(features_array)  # returns np.ndarray (n_samples, n_classes)

# Phishing detection
from cyber_ai.phishing.url_detector import predict_phishing
score = predict_phishing("http://example.com")  # returns float

# Plaintext classification
from cyber_ai.firewall.plaintext_classifier import predict_plaintext
score = predict_plaintext("GET /admin/login.php")  # returns float

# Encrypted traffic
from cyber_ai.firewall.encrypted_cnn import predict_encrypted_traffic
score = predict_encrypted_traffic(flow_features)  # returns float

# Risk scoring
from cyber_ai.risk_engine.risk_scoring import compute_risk_score
risk = compute_risk_score(
    malware_score=0.8,
    encrypted_score=0.5,
    plaintext_score=0.3,
    vt_score=0.6,
    suricata_score=0.9
)  # returns float

# Zero trust
from cyber_ai.zero_trust.access_control import evaluate_session
decision = evaluate_session("session_123", risk_score=0.82)  # returns str

# VirusTotal
from cyber_ai.intelligence.virustotal import safe_query_hash, safe_query_url
vt_score = safe_query_hash("sha256_hash_here")  # returns float
vt_url_score = safe_query_url("http://example.com")  # returns float

# Suricata
from cyber_ai.intelligence.suricata_parser import parse_suricata_logs
alerts = parse_suricata_logs("eve.json", max_alerts=5000)  # returns List[AlertRecord]
```

---

## Configuration

### Model Persistence

Trained models are automatically saved to `saved_models/` directories within each module folder:

```
malware_detection/saved_models/
  malware_rf_model.joblib
  malware_pipeline.joblib
  malware_label_encoder.joblib

phishing/saved_models/
  phishing_rf_model.joblib
  phishing_scaler.joblib

firewall/saved_models/
  plaintext_rf_model.joblib
  plaintext_tfidf.joblib
  encrypted_cnn_model.pth
  encrypted_scaler.joblib
  encrypted_cnn_config.joblib
```

On subsequent runs, models are loaded from disk instead of retraining.

### Incident Logs

Forensic snapshots are saved as JSON files to `cyber_ai/incidents/`:

```json
{
  "timestamp": "2026-03-08T21:45:00",
  "flow_key": "192.168.1.100:54321->10.0.0.5:443/TCP",
  "risk_score": 0.92,
  "decision": "Block",
  "scores": {
    "malware": 0.8,
    "encrypted": 0.9,
    "plaintext": 0.1,
    "vt": 0.7,
    "suricata": 0.5
  }
}
```

---

## Project Structure

```
cyber_ai/
├── README.md
├── requirements.txt
├── __init__.py
├── main.py
│
├── malware_detection/
│   ├── __init__.py
│   └── malware_model.py
│
├── phishing/
│   ├── __init__.py
│   └── url_detector.py
│
├── firewall/
│   ├── __init__.py
│   ├── flow_table.py
│   ├── plaintext_classifier.py
│   └── encrypted_cnn.py
│
├── intelligence/
│   ├── __init__.py
│   ├── virustotal.py
│   └── suricata_parser.py
│
├── risk_engine/
│   ├── __init__.py
│   └── risk_scoring.py
│
├── zero_trust/
│   ├── __init__.py
│   └── access_control.py
│
├── incident_response/
│   ├── __init__.py
│   └── response_engine.py
│
└── dashboard/
    ├── __init__.py
    └── app.py
```

---

## Research References

The encrypted traffic classification CNN is informed by the following research:

1. **PacketCGAN** - Class imbalance for encrypted traffic classification using CNN. [arXiv:1911.12046](https://arxiv.org/abs/1911.12046)
2. **AI-Based Network Traffic Classification** - CNN + LSTM for encrypted/obfuscated data (2025). [DOI:10.63075/4bth0029](https://doi.org/10.63075/4bth0029)
3. **Deep Learning for Encrypted Traffic** - DNN including CNN layers for encrypted flow classification. Sensors 2022, 22(19):7643.
4. **Autoencoder + CNN** - Combining autoencoders with CNNs for robust encrypted traffic analysis. [PubMed:40991620](https://pubmed.ncbi.nlm.nih.gov/40991620)

---

## License

This project is built for educational and research purposes as part of the India Innovates initiative.

---

<p align="center">
  Built with Python, scikit-learn, PyTorch, Streamlit, and real-world security data.
</p>
