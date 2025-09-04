# eval.py
"""
Evaluation + simple plots (student-friendly).

What this does:
- Loads ./outputs/best_model.pth and ./outputs/history.pkl
- Plots Training vs Validation Accuracy/Loss (saves PNGs in ./outputs and shows them)
- Evaluates on test.csv and prints test loss/accuracy
- Prints a confusion matrix with readable class names
- Shows a few sample predictions (true vs predicted) with confidence

Run:
    python eval.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import get_device, load_checkpoint, accuracy
from dataset import FERPixelCSV, build_transforms
from model import get_resnet18

# -----------------------------
# Paths (resolve relative to this file)
# -----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
TEST_CSV = os.path.join(BASE, "test.csv")
CKPT_PATH = os.path.join(OUT_DIR, "best_model.pth")
HIST_PATH = os.path.join(OUT_DIR, "history.pkl")

# -----------------------------
# Human-readable class names
# -----------------------------
IDX_TO_CLASS = {
    0: "anger",
    1: "disgust",
    2: "sad",
    3: "happiness",
    4: "surprise",
}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS.keys())]


def plot_history(history: dict, out_dir: str, show: bool = True) -> None:
    """Make and save simple Accuracy/Loss plots."""
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    acc_png = os.path.join(out_dir, "acc_curve.png")
    plt.savefig(acc_png, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_png, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved plots:\n  {acc_png}\n  {loss_png}")


def main():
    device = get_device()
    print("Device:", device)

    # 1) Load history (if present) and plot
    if os.path.isfile(HIST_PATH):
        with open(HIST_PATH, "rb") as f:
            history = pickle.load(f)
        # In notebook environments, %run shows plots inline; in !python they are saved.
        plot_history(history, OUT_DIR, show=True)
    else:
        print(f"[Info] History not found at {HIST_PATH} — skipping plots.")

    # 2) Build test loader
    if not os.path.isfile(TEST_CSV):
        raise FileNotFoundError(f"Missing test CSV at: {TEST_CSV}")

    _, eval_tf = build_transforms(image_size=224)
    test_ds = FERPixelCSV(TEST_CSV, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    num_classes = len(np.unique(test_ds.labels))
    print(f"Test samples: {len(test_ds)} | num_classes: {num_classes}")

    # 3) Create model + load trained weights
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Train first (python train.py).")

    model = get_resnet18(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    # 4) Evaluate on test set
    all_logits, all_targets = [], []
    total_loss, total_n = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

            total_loss += loss.item() * images.size(0)
            total_n += labels.numel()

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    test_loss = total_loss / max(total_n, 1)
    test_acc = accuracy(all_logits, all_targets)

    print(f"\nTest results:")
    print(f"- loss: {test_loss:.4f}")
    print(f"- acc : {test_acc:.2f}%")

    # 5) Confusion matrix with class names
    try:
        from sklearn.metrics import confusion_matrix
        import pandas as pd

        preds = all_logits.argmax(dim=1).numpy()
        targs = all_targets.numpy()
        cm = confusion_matrix(targs, preds, labels=list(range(len(CLASS_NAMES))))

        print("\nConfusion matrix (rows=true, cols=pred):")
        try:
            df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
            print(df_cm)
        except Exception:
            # Fallback to plain print if pandas formatting fails
            print(cm)
            print("Order of classes:", CLASS_NAMES)
    except Exception:
        print("\n[Note] scikit-learn or pandas not installed -> skipping named confusion matrix.")

    # 6) Show a few sample predictions with names
    probs = torch.softmax(all_logits, dim=1).numpy()
    print("\nSample predictions (first 5):")
    for i in range(min(5, len(test_ds))):
        true_idx = int(all_targets[i].item())
        pred_idx = int(np.argmax(probs[i]))
        conf = float(np.max(probs[i]) * 100.0)
        true_name = IDX_TO_CLASS.get(true_idx, str(true_idx))
        pred_name = IDX_TO_CLASS.get(pred_idx, str(pred_idx))
        print(f"  #{i:02d} | true={true_name}  pred={pred_name}  conf={conf:.1f}%")

if __name__ == "__main__":
    main()

