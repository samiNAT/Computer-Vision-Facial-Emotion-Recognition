# utils.py
"""
Simple utilities for the Facial Emotion project.

What’s inside:
- set_seed(seed): make results more repeatable
- get_device(): pick 'cuda' if available, else 'cpu'
- ensure_dir(path): create a folder if it doesn't exist
- accuracy(logits, targets): quick classification accuracy
- save_checkpoint(model, path): save model weights
- load_checkpoint(model, path, device): load model weights

Keep it simple and readable for student presentations.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds so runs are a bit more repeatable.
    (Note: complete determinism can slow things down; this is a light version.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Device selection
# -----------------------------
def get_device() -> torch.device:
    """
    Return CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Filesystem helper
# -----------------------------
def ensure_dir(path: str) -> None:
    """
    Create the folder if it doesn't exist (no error if it already exists).
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute simple top-1 accuracy for classification.
    Args:
        logits: [B, C] model outputs (before softmax).
        targets: [B] ground-truth class indices.
    Returns:
        accuracy in percentage (0-100).
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return 100.0 * correct / max(total, 1)


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """
    Save model weights only (state_dict) to a file.
    """
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: Optional[torch.device] = None) -> None:
    """
    Load model weights (state_dict) from a file.
    """
    if device is None:
        device = get_device()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
