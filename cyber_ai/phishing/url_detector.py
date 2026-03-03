"""
Phishing URL detection using lexical features + RandomForest.

Trains on real labeled URL datasets (no synthetic data). Supports:
- HuggingFace dataset `testpj/phishing-dataset` (if it contains a `url` field)
- Local CSV containing URLs and labels

Exports:
- train_url_model(...)
- predict_phishing(url_str: str) -> probability: float
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "url_rf.joblib"


@dataclass
class UrlModelConfig:
    csv_path: Optional[Path] = None
    hf_dataset: Optional[str] = "testpj/phishing-dataset"
    url_column: str = "url"
    label_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 500
    n_jobs: int = -1


def _load_from_huggingface(dataset_name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "HuggingFace `datasets` is required to load HF datasets. "
            "Install it with `pip install datasets` or provide a CSV instead."
        ) from e

    ds = load_dataset(dataset_name)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    if df.empty:
        raise ValueError(f"HuggingFace dataset {dataset_name} split '{split_name}' is empty.")
    return df


def _normalize_labels(y: pd.Series) -> np.ndarray:
    if np.issubdtype(y.dtype, np.number):
        vals = y.to_numpy()
        uniques = sorted(set(np.unique(vals).tolist()))
        if set(uniques) <= {0, 1}:
            return vals.astype(int)
        return (vals != 0).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    malicious_tokens = {"phishing", "malicious", "bad", "1", "true", "yes"}
    benign_tokens = {"benign", "legitimate", "good", "0", "false", "no"}
    mapped = []
    for v in y_str.tolist():
        if v in malicious_tokens:
            mapped.append(1)
        elif v in benign_tokens:
            mapped.append(0)
        else:
            raise ValueError(f"Unrecognized label value '{v}'. Configure mapping for your dataset.")
    return np.asarray(mapped, dtype=int)


_RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    ent = 0.0
    n = len(s)
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return float(ent)


def _url_features(url: str) -> np.ndarray:
    u = str(url).strip()
    parsed = urlparse(u if "://" in u else "http://" + u)

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = parsed.scheme or ""

    # Character counts
    features = {
        "url_len": len(u),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "num_dots": u.count("."),
        "num_hyphen": u.count("-"),
        "num_at": u.count("@"),
        "num_question": u.count("?"),
        "num_equal": u.count("="),
        "num_amp": u.count("&"),
        "num_percent": u.count("%"),
        "num_slash": u.count("/"),
        "num_colon": u.count(":"),
        "num_underscore": u.count("_"),
        "num_tilde": u.count("~"),
        "num_comma": u.count(","),
        "num_semicolon": u.count(";"),
        "num_dollar": u.count("$"),
        "num_space": u.count(" "),
        "has_https": 1 if scheme.lower() == "https" else 0,
        "has_ip_host": 1 if _RE_IP.match(host) else 0,
        "host_num_digits": sum(ch.isdigit() for ch in host),
        "url_num_digits": sum(ch.isdigit() for ch in u),
        "entropy_url": _shannon_entropy(u),
        "entropy_host": _shannon_entropy(host),
        "entropy_path": _shannon_entropy(path),
        "num_subdomains": max(0, host.count(".") - 1) if host else 0,
        "tld_len": len(host.split(".")[-1]) if "." in host else 0,
    }

    return np.asarray(list(features.values()), dtype=float)


def build_feature_matrix(urls: pd.Series) -> Tuple[np.ndarray, list]:
    feat_list = [_url_features(u) for u in urls.astype(str).tolist()]
    X = np.vstack(feat_list)
    feature_names = [
        "url_len",
        "host_len",
        "path_len",
        "query_len",
        "num_dots",
        "num_hyphen",
        "num_at",
        "num_question",
        "num_equal",
        "num_amp",
        "num_percent",
        "num_slash",
        "num_colon",
        "num_underscore",
        "num_tilde",
        "num_comma",
        "num_semicolon",
        "num_dollar",
        "num_space",
        "has_https",
        "has_ip_host",
        "host_num_digits",
        "url_num_digits",
        "entropy_url",
        "entropy_host",
        "entropy_path",
        "num_subdomains",
        "tld_len",
    ]
    return X, feature_names


def load_url_dataset(config: UrlModelConfig) -> Tuple[pd.Series, np.ndarray, str]:
    if config.csv_path is not None:
        if not config.csv_path.exists():
            raise FileNotFoundError(
                f"URL dataset CSV not found at {config.csv_path}. "
                "Provide the real dataset file; no synthetic data is generated."
            )
        df = pd.read_csv(config.csv_path)
        source = str(config.csv_path)
    else:
        if not config.hf_dataset:
            raise ValueError("Either csv_path or hf_dataset must be provided.")
        df = _load_from_huggingface(config.hf_dataset)
        source = f"hf:{config.hf_dataset}"

    if config.url_column not in df.columns:
        raise ValueError(
            f"URL column '{config.url_column}' not found in dataset ({source}). "
            f"Available columns: {list(df.columns)}"
        )
    if config.label_column not in df.columns:
        raise ValueError(
            f"Label column '{config.label_column}' not found in dataset ({source}). "
            f"Available columns: {list(df.columns)}"
        )

    urls = df[config.url_column].astype(str)
    y = _normalize_labels(df[config.label_column])
    return urls, y, source


def train_url_model(config: UrlModelConfig, model_output_path: Path = DEFAULT_MODEL_PATH) -> dict:
    urls, y, source = load_url_dataset(config)
    X, feature_names = build_feature_matrix(urls)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "source": source,
        "feature_names": feature_names,
    }

    joblib.dump({"model": clf, "feature_names": feature_names, "config": config.__dict__}, model_output_path)
    with model_output_path.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _load_url_model(model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[RandomForestClassifier, list]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"URL phishing model not found at {model_path}. Train it first with train_url_model."
        )
    payload = joblib.load(model_path)
    return payload["model"], payload["feature_names"]


def predict_phishing(url_str: str, model_path: Optional[Path] = None) -> float:
    mp = model_path or DEFAULT_MODEL_PATH
    model, feature_names = _load_url_model(mp)
    x = _url_features(url_str).reshape(1, -1)
    if x.shape[1] != len(feature_names):
        raise ValueError("Internal feature size mismatch; retrain the URL model.")
    proba = model.predict_proba(x)[0, 1]
    return float(proba)


if __name__ == "__main__":
    """
    Example CLI entry point to train the URL phishing model.

    Options:
    - Set PHISHING_URL_CSV_PATH to train from a local CSV
    - Otherwise it will attempt to download HuggingFace dataset testpj/phishing-dataset

    You may also set:
    - PHISHING_URL_COLUMN (default: url)
    - PHISHING_LABEL_COLUMN (default: label)
    """
    csv_path = os.getenv("PHISHING_URL_CSV_PATH")
    cfg = UrlModelConfig(
        csv_path=Path(csv_path) if csv_path else None,
        hf_dataset=os.getenv("PHISHING_HF_DATASET", "testpj/phishing-dataset"),
        url_column=os.getenv("PHISHING_URL_COLUMN", "url"),
        label_column=os.getenv("PHISHING_LABEL_COLUMN", "label"),
    )
    metrics = train_url_model(cfg)
    print(json.dumps(metrics, indent=2))

