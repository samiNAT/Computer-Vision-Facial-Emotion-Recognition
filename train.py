
# train.py
"""
Simple training script for Facial Emotion Recognition (CSV pixel dataset).

What this does:
- Loads train/val/test CSVs (found in the SAME folder as this file)
- Builds a ResNet18 model with 5-class output
- Trains for a few epochs on CPU (or GPU if available)
- Prints progress inside each epoch (avg loss/acc)
- Saves the best model to ./outputs/best_model.pth
- Saves training history to ./outputs/history.pkl  (for plotting later)

New:
- CLI: --epochs N (default 30)
- CLI: --stop_acc THRESH  -> if provided, auto-stops once val_acc >= THRESH
"""

import os
import pickle
import argparse  # <-- added for CLI args
import torch
import torch.nn as nn
import torch.optim as optim

from utils import set_seed, get_device, ensure_dir, save_checkpoint
from dataset import get_dataloaders
from model import get_resnet18


def train_one_epoch(model, loader, device, optimizer, criterion, epoch: int):
    """
    Run a single training epoch over the dataset.
    Returns (epoch_loss, epoch_acc).
    """
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.numel()

        # show running averages every 50 batches
        if batch_idx % 50 == 0:
            avg_loss = running_loss / max(running_total, 1)
            avg_acc = 100.0 * running_correct / max(running_total, 1)
            print(f"[Epoch {epoch:02d}] Batch {batch_idx}/{len(loader)} "
                  f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.2f}%")

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = 100.0 * running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, device, criterion):
    """Validation pass (no gradients) that returns (loss, accuracy)."""
    model.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.numel()

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = 100.0 * running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


def parse_args():
    """
    CLI options:
      --epochs N       number of epochs (default: 30)
      --stop_acc X     if set, stop once val_acc >= X (percentage, e.g., 85.0)
    """
    p = argparse.ArgumentParser(description="Train Facial Emotion Recognition")
    p.add_argument("--epochs", type=int, default=30,
                   help="number of training epochs (default: 30)")
    p.add_argument("--stop_acc", type=float, default=None,
                   help="early stop when validation accuracy (%%) >= this threshold; disabled by default")
    return p.parse_args()


def main():
    # --- setup
    args = parse_args()
    set_seed(42)
    device = get_device()
    BASE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(BASE, "outputs")
    ensure_dir(OUT)
    print("Using device:", device)
    print(f"Config -> epochs: {args.epochs} | stop_acc: {args.stop_acc}")

    # --- paths (relative to this file)
    train_csv = os.path.join(BASE, "train.csv")
    val_csv   = os.path.join(BASE, "val.csv")
    test_csv  = os.path.join(BASE, "test.csv")

    print("Resolved paths:")
    for pth in (train_csv, val_csv, test_csv):
        print("  ", pth, "exists:", os.path.exists(pth))

    # --- data
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        train_csv, val_csv, test_csv,
        image_size=224, batch_size=32, num_workers=2,  # 0 for Windows
        use_weighted_sampler=True
    )
    num_classes = meta["num_classes"]
    print("Data ready:", meta)

    # --- model/opt/loss
    model = get_resnet18(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # --- training config
    epochs = int(args.epochs)  # default 30 if not provided
    best_val_acc = 0.0
    best_path = os.path.join(OUT, "best_model.pth")

    # --- history for plotting later
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc:.2f}%")

        # record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # save best by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_path)
            print(f"  -> saved new best model to {best_path} (val_acc={best_val_acc:.2f}%)")

        # optional early stop by accuracy threshold (percentage)
        if args.stop_acc is not None and val_acc >= args.stop_acc:
            print(f"[Early Stop] Reached threshold: val_acc={val_acc:.2f}% >= {args.stop_acc:.2f}%. Stopping.")
            break

    # save history for eval plotting
    hist_path = os.path.join(OUT, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print("\nTraining done. Best val acc:", f"{best_val_acc:.2f}%")
    print(f"History saved to {hist_path}")
    print("Now you can run:  python ./eval.py")


if __name__ == "__main__":
    main()
