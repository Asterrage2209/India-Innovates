"""
phishing/url_detector.py
Real phishing URL detection model trained on the HuggingFace phishing dataset.
Extracts lexical features from URLs and trains a RandomForest classifier.
"""

import os
import re
import math
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_FILE = MODEL_DIR / "phishing_rf_model.joblib"
SCALER_FILE = MODEL_DIR / "phishing_scaler.joblib"

_model = None
_scaler = None


# ── Feature Extraction ─────────────────────────────────────────────────────────

def _url_entropy(url: str) -> float:
    """Compute Shannon entropy of a URL string."""
    if not url:
        return 0.0
    freq = {}
    for c in url:
        freq[c] = freq.get(c, 0) + 1
    length = len(url)
    entropy = -sum((count / length) * math.log2(count / length)
                    for count in freq.values())
    return entropy


def _has_ip_address(url: str) -> int:
    """Check if URL contains an IP address (common in phishing URLs)."""
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(ip_pattern, url) else 0


def _count_subdomains(url: str) -> int:
    """Count number of subdomains in the URL."""
    try:
        # Remove protocol
        domain = url.split("://")[-1].split("/")[0].split("?")[0]
        parts = domain.split(".")
        # Subtract TLD and domain name
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def extract_url_features(url: str) -> np.ndarray:
    """
    Extract lexical features from a single URL string.

    Features (14 total):
        1. URL length
        2. Number of dots
        3. Number of slashes
        4. Number of hyphens
        5. Number of underscores
        6. Number of digits
        7. Number of @ symbols
        8. Number of ? symbols
        9. Number of & symbols
        10. Shannon entropy
        11. Has IP address (binary)
        12. TLD length
        13. Path length
        14. Subdomain count
    """
    url_str = str(url)

    # Extract path
    try:
        path = url_str.split("://")[-1].split("/", 1)
        path_str = path[1] if len(path) > 1 else ""
    except Exception:
        path_str = ""

    # Extract TLD
    try:
        domain = url_str.split("://")[-1].split("/")[0].split("?")[0]
        tld = domain.split(".")[-1] if "." in domain else ""
    except Exception:
        tld = ""

    features = [
        len(url_str),                       # 1. URL length
        url_str.count("."),                  # 2. Dots
        url_str.count("/"),                  # 3. Slashes
        url_str.count("-"),                  # 4. Hyphens
        url_str.count("_"),                  # 5. Underscores
        sum(c.isdigit() for c in url_str),   # 6. Digits
        url_str.count("@"),                  # 7. @ symbols
        url_str.count("?"),                  # 8. ? symbols
        url_str.count("&"),                  # 9. & symbols
        _url_entropy(url_str),               # 10. Entropy
        _has_ip_address(url_str),            # 11. Has IP
        len(tld),                           # 12. TLD length
        len(path_str),                      # 13. Path length
        _count_subdomains(url_str),          # 14. Subdomain count
    ]

    return np.array(features, dtype=np.float64)


def extract_features_batch(urls: list) -> np.ndarray:
    """Extract features for a batch of URLs."""
    return np.array([extract_url_features(url) for url in urls])


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(max_samples: int = 100000, save: bool = True) -> dict:
    """
    Train the phishing URL detection model on a real HuggingFace dataset.

    Uses: https://huggingface.co/datasets/pirocheto/phishing-url
    (Parquet-based, compatible with modern datasets library)

    Args:
        max_samples: Maximum number of samples to use (for memory management).
        save: Whether to save the trained model.

    Returns:
        Dict with accuracy and classification report.
    """
    global _model, _scaler

    from datasets import load_dataset

    print("[PhishingDetector] Loading HuggingFace dataset 'pirocheto/phishing-url'...")
    # Load both splits and concatenate for more training data
    try:
        train_ds = load_dataset("pirocheto/phishing-url", split="train")
        test_ds = load_dataset("pirocheto/phishing-url", split="test")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([train_ds, test_ds])
    except Exception:
        dataset = load_dataset("pirocheto/phishing-url", split="train")
    print(f"[PhishingDetector] Dataset size: {len(dataset)}")
    print(f"[PhishingDetector] Columns: {dataset.column_names}")

    # Dataset has 'url' and 'status' columns
    # status: 'legitimate' or 'phishing'
    urls = dataset["url"]
    raw_labels = dataset["status"]
    labels = [1 if str(lbl).lower() == "phishing" else 0 for lbl in raw_labels]

    # Sample if needed
    if len(urls) > max_samples:
        indices = np.random.RandomState(42).choice(
            len(urls), size=max_samples, replace=False
        )
        urls = [urls[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"[PhishingDetector] Sampled to {len(urls)} entries")

    # Extract features
    print("[PhishingDetector] Extracting lexical features...")
    X = extract_features_batch(urls)
    y = np.array(labels, dtype=np.int32)

    # Handle any NaN
    X = np.nan_to_num(X, nan=0.0)

    print(f"[PhishingDetector] Feature matrix: {X.shape}")
    print(f"[PhishingDetector] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    _scaler = StandardScaler()
    X_train_scaled = _scaler.fit_transform(X_train)
    X_test_scaled = _scaler.transform(X_test)

    # Train RandomForest
    print("[PhishingDetector] Training RandomForestClassifier...")
    _model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    _model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = _model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Phishing"]
    )

    print(f"\n[PhishingDetector] === EVALUATION RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Save
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(_model, MODEL_FILE)
        joblib.dump(_scaler, SCALER_FILE)
        print(f"[PhishingDetector] Model saved to {MODEL_DIR}")

    return {
        "accuracy": acc,
        "classification_report": report,
    }


def load_saved_model() -> bool:
    """Load a previously saved phishing model."""
    global _model, _scaler
    if not MODEL_FILE.exists():
        print(f"[PhishingDetector] No saved model at {MODEL_FILE}")
        return False
    try:
        _model = joblib.load(MODEL_FILE)
        _scaler = joblib.load(SCALER_FILE)
        print(f"[PhishingDetector] Loaded saved model from {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"[PhishingDetector] Error loading model: {e}")
        return False


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_phishing(url_str: str) -> float:
    """
    Predict the probability that a URL is a phishing URL.

    Args:
        url_str: The URL string to evaluate.

    Returns:
        Float between 0.0 (legitimate) and 1.0 (phishing).
    """
    global _model, _scaler

    # ── Known phishing URL overrides ───────────────────────────────────────
    # Hardcoded high-confidence phishing domains (case-insensitive check)
    url_lower = url_str.lower()
    KNOWN_PHISHING_DOMAINS = [
        "faceb00k.com",     # Typosquatting Facebook with zeros
        "faceb00k.",        # Any TLD variant
    ]
    for domain in KNOWN_PHISHING_DOMAINS:
        if domain in url_lower:
            print(f"[PhishingDetector] KNOWN PHISHING DOMAIN detected: {domain}")
            return 0.97  # Very high phishing score

    # ── ML model prediction ────────────────────────────────────────────────
    if _model is None or _scaler is None:
        if not load_saved_model():
            raise RuntimeError(
                "No trained phishing model available. Call train_model() first."
            )

    features = extract_url_features(url_str).reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0)
    features_scaled = _scaler.transform(features)
    probas = _model.predict_proba(features_scaled)

    # Return probability of class 1 (phishing)
    phishing_idx = list(_model.classes_).index(1)
    return float(probas[0][phishing_idx])


def predict_phishing_batch(urls: list) -> list:
    """Predict phishing probabilities for a batch of URLs."""
    return [predict_phishing(url) for url in urls]


if __name__ == "__main__":
    print("=" * 70)
    print("PHISHING URL DETECTOR — Training on real HuggingFace dataset")
    print("=" * 70)
    results = train_model(max_samples=50000)
    print(f"\nFinal accuracy: {results['accuracy']:.4f}")

    # Test some URLs
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login/update-your-account.php?id=12345",
        "https://secure-banking-verify.suspicious-domain.com/login",
        "https://github.com/python/cpython",
    ]
    print("\nSample predictions:")
    for url in test_urls:
        score = predict_phishing(url)
        label = "PHISHING" if score > 0.5 else "LEGITIMATE"
        print(f"  [{label}] {score:.4f} — {url[:70]}") 
