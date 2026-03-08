"""
firewall/encrypted_cnn.py
CNN classifier for encrypted traffic classification.
Designed for the Mendeley Encrypted Malicious Traffic Dataset.
Uses 1D-CNN in PyTorch to classify encrypted flows as malicious vs benign.

Dataset: https://data.mendeley.com/datasets/xw7r4tt54g/1

If the dataset is not available locally, the module provides clear instructions
and falls back gracefully.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_FILE = MODEL_DIR / "encrypted_cnn_model.pth"
SCALER_FILE = MODEL_DIR / "encrypted_scaler.joblib"

# Default location for the Mendeley dataset
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "encrypted_traffic"

_model = None
_scaler = None
_device = None


def _get_device():
    """Get the PyTorch device (GPU if available, else CPU)."""
    import torch
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ── CNN Architecture ───────────────────────────────────────────────────────────

def _build_cnn_model(input_size: int, num_classes: int = 2):
    """
    Build a 1D-CNN model for encrypted traffic classification.

    Architecture based on research:
    - PacketCGAN (arXiv:1911.12046): CNN on encrypted traffic flows
    - Deep Learning for Encrypted Traffic Classification (Sensors 2022)
    - Encrypted Traffic Classification with Autoencoders + CNN (PubMed 2024)

    Network:
        Conv1d(1, 32, k=3) → ReLU → MaxPool(2)
        Conv1d(32, 64, k=3) → ReLU → MaxPool(2)
        Conv1d(64, 128, k=3) → ReLU → AdaptiveAvgPool(1)
        FC(128, 64) → ReLU → Dropout(0.3)
        FC(64, num_classes)
    """
    import torch
    import torch.nn as nn

    class EncryptedTrafficCNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x shape: (batch, 1, seq_len)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return EncryptedTrafficCNN(input_size, num_classes)


# ── Data Loading ───────────────────────────────────────────────────────────────

def _find_dataset(data_dir: str = None) -> Optional[str]:
    """
    Locate the encrypted traffic dataset CSV file.

    The Mendeley dataset (xw7r4tt54g) contains CSV files with
    network flow features and labels.
    """
    search_dirs = [
        data_dir,
        str(DEFAULT_DATA_DIR),
        str(Path(__file__).resolve().parent.parent.parent / "data"),
        str(Path.home() / "Downloads" / "encrypted_traffic"),
    ]

    for d in search_dirs:
        if d is None or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".csv"):
                return os.path.join(d, f)

    return None


def load_dataset(data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the encrypted traffic dataset.

    Expected format: CSV with numeric features and a label column
    (last column or column named 'label', 'Label', or 'class').

    Returns:
        Tuple of (features, labels) numpy arrays.
    """
    if data_path is None:
        data_path = _find_dataset()

    if data_path is None:
        raise FileNotFoundError(
            "Encrypted traffic dataset not found.\n"
            "Please download from: https://data.mendeley.com/datasets/xw7r4tt54g/1\n"
            f"Extract CSV files to: {DEFAULT_DATA_DIR}\n"
            "Or pass the CSV path directly to load_dataset(data_path=...)"
        )

    print(f"[EncryptedCNN] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[EncryptedCNN] Dataset shape: {df.shape}")
    print(f"[EncryptedCNN] Columns: {list(df.columns[:10])}...")

    # Identify label column
    label_col = None
    for col_name in ["label", "Label", "class", "Class", "target", "Target"]:
        if col_name in df.columns:
            label_col = col_name
            break

    if label_col is None:
        # Assume last column is the label
        label_col = df.columns[-1]
        print(f"[EncryptedCNN] Using last column as label: '{label_col}'")

    y = df[label_col].values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values

    # Convert labels to binary if needed
    unique_labels = np.unique(y)
    print(f"[EncryptedCNN] Unique labels: {unique_labels}")

    if len(unique_labels) == 2:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif not np.issubdtype(y.dtype, np.integer):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    y = y.astype(np.int64)

    print(f"[EncryptedCNN] Features: {X.shape}, Labels: {y.shape}")
    print(f"[EncryptedCNN] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(
    data_path: str = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save: bool = True,
) -> dict:
    """
    Train the encrypted traffic CNN on the real Mendeley dataset.

    Returns:
        Dict with accuracy and loss history.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    global _model, _scaler

    X, y = load_dataset(data_path)

    # Scale features
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors — reshape for Conv1d: (batch, 1, features)
    device = _get_device()
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Build model
    num_classes = len(np.unique(y))
    model = _build_cnn_model(X_scaled.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"[EncryptedCNN] Training on {device} for {epochs} epochs...")
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
                test_acc = accuracy_score(y_test, test_preds)
                history["accuracy"].append(test_acc)
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
                      f"Test Acc: {test_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        final_acc = accuracy_score(y_test, test_preds)
        report = classification_report(
            y_test, test_preds,
            target_names=["Benign", "Malicious"][:num_classes]
        )

    print(f"\n[EncryptedCNN] === FINAL EVALUATION ===")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    _model = model

    # Save
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_FILE)
        joblib.dump(_scaler, SCALER_FILE)
        # Save model config
        joblib.dump({"input_size": X_scaled.shape[1], "num_classes": num_classes},
                    MODEL_DIR / "encrypted_cnn_config.joblib")
        print(f"[EncryptedCNN] Model saved to {MODEL_DIR}")

    return {
        "accuracy": final_acc,
        "classification_report": report,
        "history": history,
    }


def load_saved_model() -> bool:
    """Load a previously saved CNN model."""
    import torch
    global _model, _scaler

    if not MODEL_FILE.exists():
        print(f"[EncryptedCNN] No saved model at {MODEL_FILE}")
        return False

    try:
        config = joblib.load(MODEL_DIR / "encrypted_cnn_config.joblib")
        _model = _build_cnn_model(config["input_size"], config["num_classes"])
        _model.load_state_dict(torch.load(MODEL_FILE, map_location=_get_device()))
        _model.to(_get_device())
        _model.eval()
        _scaler = joblib.load(SCALER_FILE)
        print(f"[EncryptedCNN] Loaded saved model from {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"[EncryptedCNN] Error loading model: {e}")
        return False


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_encrypted_traffic(features: np.ndarray) -> float:
    """
    Predict the probability that an encrypted traffic flow is malicious.

    Args:
        features: 1D numpy array of network flow features.

    Returns:
        Float between 0.0 (benign) and 1.0 (malicious).
    """
    import torch

    global _model, _scaler

    if _model is None or _scaler is None:
        if not load_saved_model():
            print("[EncryptedCNN] WARNING: No trained model. Returning 0.0")
            print("  Download dataset from: https://data.mendeley.com/datasets/xw7r4tt54g/1")
            return 0.0

    if features.ndim == 1:
        features = features.reshape(1, -1)

    device = _get_device()
    features_scaled = _scaler.transform(features.astype(np.float32))
    x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    _model.eval()
    with torch.no_grad():
        outputs = _model(x)
        probs = torch.softmax(outputs, dim=1)
        # Return malicious probability (class 1)
        mal_prob = probs[0, min(1, probs.shape[1] - 1)].item()

    return float(mal_prob)


if __name__ == "__main__":
    print("=" * 70)
    print("ENCRYPTED TRAFFIC CNN — Training on real Mendeley dataset")
    print("=" * 70)

    try:
        results = train_model()
        print(f"\nFinal accuracy: {results['accuracy']:.4f}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nThe encrypted CNN module is ready — it needs the Mendeley dataset to train.")
        print("Once downloaded, run: python -m cyber_ai.firewall.encrypted_cnn")
