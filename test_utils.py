# test_utils.py
from utils import set_seed, get_device, ensure_dir, accuracy, save_checkpoint, load_checkpoint
import torch
import torch.nn as nn

# --- test seed ---
set_seed(42)
print("Seed set successfully.")

# --- test device ---
device = get_device()
print("Device detected:", device)

# --- test ensure_dir ---
ensure_dir("./outputs/test_dir")
print("Directory check: outputs/test_dir created (or already existed).")

# --- test accuracy ---
logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])   # pretend model outputs
targets = torch.tensor([1, 0])                    # true labels
acc = accuracy(logits, targets)
print("Accuracy test:", acc, "% (expect 100%)")

# --- test checkpoint save/load ---
model = nn.Linear(10, 2)   # a simple tiny model
save_checkpoint(model, "./outputs/test_model.pth")
print("Model saved to ./outputs/test_model.pth")

# change weights to random
for p in model.parameters():
    p.data = torch.randn_like(p.data)

# reload checkpoint
load_checkpoint(model, "./outputs/test_model.pth", device)
print("Model reloaded successfully.")
