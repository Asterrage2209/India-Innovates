"""
Encrypted traffic classification using a CNN model.

This module uses the real *Encrypted Traffic Feature Dataset for Machine Learning
and Deep Learning based Encrypted Traffic Analysis* from Mendeley
(DOI: 10.17632/xw7r4tt54g.1).

The dataset provides CSV feature files at packet and session level with 305
features and balanced malicious vs. legitimate encrypted traffic. This module
expects that you have manually downloaded the dataset from Mendeley and placed
the relevant CSV files on disk.

No synthetic data is generated. If the dataset files are missing, an informative
exception is raised and you must supply the real CSV files.

The CNN design is inspired by:
- PacketCGAN (Wang et al., 2019): packet-level CNN for encrypted traffic.
- Deep Learning for Encrypted Traffic Classification and Unknown Data Detection
  (Sensors 2022, 22(19):7643): DNN/CNN on windowed flow statistics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "encrypted_cnn.pt"


@dataclass
class EncryptedCnnConfig:
    csv_path: Path
    label_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 256
    num_epochs: int = 10
    lr: float = 1e-3


class EncryptedTrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        # Interpret the 1D feature vector as a "sequence" for 1D CNN
        x = x[None, :]  # shape (1, num_features)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y)


class EncryptedCNN(nn.Module):
    """
    Simple 1D CNN over feature sequences, following patterns from PacketCGAN /
    DeepPacket-like architectures but without synthetic data generation.
    """

    def __init__(self, num_features: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # After two pools, sequence length is roughly num_features / 4
        self._feature_len = (num_features + 3) // 4

        self.fc1 = nn.Linear(64 * self._feature_len, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _validate_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Encrypted traffic CSV not found at {path}. "
            "Download the real dataset from Mendeley (DOI: 10.17632/xw7r4tt54g.1) "
            "and point ENCRYPTED_TRAFFIC_CSV_PATH to the chosen CSV file."
        )
    return pd.read_csv(path)


def load_encrypted_dataset(config: EncryptedCnnConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the encrypted traffic dataset from the provided CSV.

    The CSV must contain:
    - A binary label column (default: 'label'), where malicious/benign are
      encoded numerically (e.g., 1 for malicious, 0 for benign) or as strings
      that can be factorized.
    - Numeric feature columns derived from the 305 traffic features.
    """
    df = _validate_csv(config.csv_path)
    if config.label_column not in df.columns:
        raise ValueError(
            f"Label column '{config.label_column}' not found in CSV. "
            "Inspect the downloaded dataset and adjust EncryptedCnnConfig accordingly."
        )

    y_raw = df[config.label_column]
    if not np.issubdtype(y_raw.dtype, np.number):
        y, uniques = pd.factorize(y_raw)
    else:
        y = y_raw.to_numpy()

    feature_cols = [c for c in df.columns if c != config.label_column and df[c].dtype != "O"]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for encrypted traffic dataset.")

    X = df[feature_cols].fillna(0.0).to_numpy()
    return X, y


def train_encrypted_cnn(
    config: EncryptedCnnConfig,
    model_output_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Train the CNN on the real encrypted traffic dataset and persist model.
    Returns a dict with basic evaluation metrics.
    """
    X, y = load_encrypted_dataset(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    num_features = X.shape[1]
    num_classes = int(np.unique(y).shape[0])

    train_ds = EncryptedTrafficDataset(X_train, y_train)
    test_ds = EncryptedTrafficDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = EncryptedCNN(num_features=num_features, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                test_loss += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        test_loss /= len(test_ds)
        test_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["test_accuracy"].append(float(test_acc))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "num_classes": num_classes,
            "config": config.__dict__,
            "history": history,
        },
        model_output_path,
    )

    metrics_path = model_output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


def _load_trained_cnn(model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[EncryptedCNN, int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Encrypted CNN model not found at {model_path}. "
            "Train it first by calling train_encrypted_cnn with the real dataset."
        )

    payload = torch.load(model_path, map_location=DEVICE)
    num_features = int(payload["num_features"])
    num_classes = int(payload["num_classes"])
    model = EncryptedCNN(num_features=num_features, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, num_features


def predict_encrypted_traffic(features: np.ndarray, model_path: Optional[Path] = None) -> float:
    """
    Predict the probability that a single encrypted flow is malicious.

    Parameters
    ----------
    features : np.ndarray
        1D array with the same feature dimensionality used during training.
    model_path : Optional[Path]
        Path to the saved CNN model. If omitted, uses the default model path.

    Returns
    -------
    probability : float
        Real-valued probability in [0, 1] that the flow is malicious.
        This is derived from the softmax output of the CNN.
    """
    mp = model_path or DEFAULT_MODEL_PATH
    model, num_features = _load_trained_cnn(mp)

    feat = np.asarray(features, dtype=np.float32)
    if feat.ndim != 1:
        raise ValueError("predict_encrypted_traffic expects a 1D feature vector for a single flow.")
    if feat.shape[0] != num_features:
        raise ValueError(
            f"Feature length mismatch: model expects {num_features} features, "
            f"but received vector of length {feat.shape[0]}."
        )

    x = torch.from_numpy(feat[None, None, :]).to(DEVICE)  # (1, 1, num_features)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Assume class index 1 corresponds to malicious; if labels were reversed
    # during training, the user should map accordingly based on the dataset.
    if probs.shape[0] == 1:
        malicious_prob = float(probs[0])
    else:
        malicious_prob = float(probs[1])
    return malicious_prob


if __name__ == "__main__":
    """
    Example CLI entry point to train the encrypted CNN model.

    You MUST set:
    - ENCRYPTED_TRAFFIC_CSV_PATH: path to a real CSV file from the Mendeley dataset.
    - Optionally ENCRYPTED_LABEL_COLUMN if the label column is not named 'label'.
    """
    csv_path = os.getenv("ENCRYPTED_TRAFFIC_CSV_PATH")
    if not csv_path:
        raise SystemExit(
            "ENCRYPTED_TRAFFIC_CSV_PATH environment variable must be set to a real CSV "
            "file from the encrypted traffic dataset."
        )
    label_col = os.getenv("ENCRYPTED_LABEL_COLUMN", "label")
    cfg = EncryptedCnnConfig(csv_path=Path(csv_path), label_column=label_col)
    history = train_encrypted_cnn(cfg)
    print("Encrypted CNN trained with real data. Training history:")
    print(json.dumps(history, indent=2))

