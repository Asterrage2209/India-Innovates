"""
Plaintext payload classification for phishing/social engineering signals.

This module trains a real classifier on a real dataset (no synthetic data).
It supports loading phishing datasets from:
- HuggingFace dataset `testpj/phishing-dataset` (downloaded via `datasets` library)
- A local CSV with labeled text/URL rows

Model:
- TF-IDF vectorizer + RandomForestClassifier (scikit-learn Pipeline)

Exports:
- train_plaintext_classifier(...)
- predict_plaintext(payload_str: str) -> probability: float
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "plaintext_rf.joblib"


@dataclass
class PlaintextConfig:
    # Choose ONE of:
    csv_path: Optional[Path] = None
    hf_dataset: Optional[str] = "testpj/phishing-dataset"

    # Column configuration for CSV:
    text_column: str = "text"
    url_column_fallback: str = "url"
    label_column: str = "label"

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    n_jobs: int = -1
    max_features: int = 200_000


def _normalize_label_series(y: pd.Series) -> Tuple[np.ndarray, dict]:
    """
    Normalize labels into {0,1} with 1 meaning phishing/malicious.
    """
    if np.issubdtype(y.dtype, np.number):
        vals = y.to_numpy()
        uniques = sorted(set(np.unique(vals).tolist()))
        if set(uniques) <= {0, 1}:
            return vals.astype(int), {"0": 0, "1": 1}
        # If numeric but not binary, treat non-zero as malicious.
        return (vals != 0).astype(int), {"nonzero": 1, "zero": 0}

    y_str = y.astype(str).str.strip().str.lower()
    # common variants across public phishing datasets
    malicious_tokens = {"phishing", "malicious", "bad", "1", "true", "yes"}
    benign_tokens = {"benign", "legitimate", "good", "0", "false", "no"}

    mapped = []
    for v in y_str.tolist():
        if v in malicious_tokens:
            mapped.append(1)
        elif v in benign_tokens:
            mapped.append(0)
        else:
            raise ValueError(
                f"Unrecognized label value '{v}'. Configure label mapping for your dataset."
            )
    return np.asarray(mapped, dtype=int), {"malicious": 1, "benign": 0}


def _load_from_huggingface(dataset_name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "HuggingFace `datasets` is required to load HF datasets. "
            "Install it with `pip install datasets` or provide a CSV instead."
        ) from e

    ds = load_dataset(dataset_name)
    # pick a split: prefer 'train', otherwise first available
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    if df.empty:
        raise ValueError(f"HuggingFace dataset {dataset_name} split '{split_name}' is empty.")
    return df


def load_plaintext_dataset(config: PlaintextConfig) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Load dataset and return (texts, labels, meta).

    If the dataset doesn't have a dedicated text field, URLs can be used as
    text input (still real data, just different modality).
    """
    if config.csv_path is not None:
        if not config.csv_path.exists():
            raise FileNotFoundError(
                f"Plaintext phishing CSV not found at {config.csv_path}. "
                "Provide the real dataset file; no synthetic data is generated."
            )
        df = pd.read_csv(config.csv_path)
        source = str(config.csv_path)
    else:
        if not config.hf_dataset:
            raise ValueError("Either csv_path or hf_dataset must be provided.")
        df = _load_from_huggingface(config.hf_dataset)
        source = f"hf:{config.hf_dataset}"

    if config.label_column not in df.columns:
        raise ValueError(
            f"Label column '{config.label_column}' not found in dataset ({source}). "
            f"Available columns: {list(df.columns)}"
        )

    text_col = None
    if config.text_column in df.columns:
        text_col = config.text_column
    elif config.url_column_fallback in df.columns:
        text_col = config.url_column_fallback
    else:
        raise ValueError(
            f"Neither '{config.text_column}' nor '{config.url_column_fallback}' found in dataset ({source}). "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_col].astype(str)
    y, label_map = _normalize_label_series(df[config.label_column])

    meta = {"source": source, "text_column": text_col, "label_column": config.label_column, "label_map": label_map}
    return texts, pd.Series(y), meta


def train_plaintext_classifier(config: PlaintextConfig, model_output_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """
    Train TF-IDF + RandomForest classifier and save model + metrics.
    """
    texts, y, meta = load_plaintext_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b",
                    max_features=config.max_features,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    n_jobs=config.n_jobs,
                    random_state=config.random_state,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "meta": meta,
    }

    joblib.dump({"pipeline": pipeline, "config": config.__dict__, "meta": meta}, model_output_path)
    with model_output_path.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _load_plaintext_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Plaintext classifier not found at {model_path}. Train it first with train_plaintext_classifier."
        )
    payload = joblib.load(model_path)
    return payload["pipeline"]


def predict_plaintext(payload_str: str, model_path: Optional[Path] = None) -> float:
    """
    Return the model probability of phishing/malicious for a plaintext payload.
    """
    mp = model_path or DEFAULT_MODEL_PATH
    pipeline = _load_plaintext_model(mp)
    proba = pipeline.predict_proba([str(payload_str)])[:, 1]
    return float(proba[0])


if __name__ == "__main__":
    """
    Example CLI entry point to train plaintext classifier.

    Options:
    - Set PLAINTEXT_PHISHING_CSV_PATH to train from a local CSV
    - Otherwise it will attempt to download HuggingFace dataset testpj/phishing-dataset

    You may also set:
    - PLAINTEXT_TEXT_COLUMN (default: text)
    - PLAINTEXT_LABEL_COLUMN (default: label)
    - PLAINTEXT_URL_COLUMN_FALLBACK (default: url)
    """
    csv_path = os.getenv("PLAINTEXT_PHISHING_CSV_PATH")
    cfg = PlaintextConfig(
        csv_path=Path(csv_path) if csv_path else None,
        hf_dataset=os.getenv("PLAINTEXT_HF_DATASET", "testpj/phishing-dataset"),
        text_column=os.getenv("PLAINTEXT_TEXT_COLUMN", "text"),
        label_column=os.getenv("PLAINTEXT_LABEL_COLUMN", "label"),
        url_column_fallback=os.getenv("PLAINTEXT_URL_COLUMN_FALLBACK", "url"),
    )
    out = train_plaintext_classifier(cfg)
    print(json.dumps(out, indent=2))

