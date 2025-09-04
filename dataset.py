# dataset.py
"""
Dataset + DataLoader utilities for Facial Emotion Recognition (CSV with pixel strings).

Main components:
---------------
- FERPixelCSV: 
    A custom PyTorch Dataset that reads a CSV file containing image data in 
    pixel-string format and labels for facial emotions.
    The CSV rows are expected to have a label column and a pixel column.

- build_transforms: 
    Returns torchvision transform pipelines for training and evaluation (resize, crop, 
    normalization, augmentation).

- get_dataloaders: 
    Utility to create train/val/test DataLoaders. 
    Can also build a WeightedRandomSampler to handle class imbalance.

Run directly (python dataset.py):
--------------------------------
Will perform a "smoke test" to verify that train/val/test DataLoaders 
load correctly from CSV files located in the same folder.
"""

import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# __all__ defines which symbols are exported when "from dataset import *" is used
__all__ = ["FERPixelCSV", "build_transforms", "get_dataloaders"]


# =========================================
# Constants (ImageNet normalization)
# =========================================
# Precomputed mean and std of ImageNet dataset. 
# Using them here allows compatibility with pretrained CNN backbones (e.g. ResNet18).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =========================================
# Helper functions
# =========================================
def _detect_square_side(n_pixels: int) -> int:
    """
    Given a total pixel count, return the side length if it forms a perfect square.
    Example: 48*48 = 2304 pixels → returns 48.
    """
    side = int(math.sqrt(n_pixels))
    if side * side != n_pixels:
        raise ValueError(f"Pixel count {n_pixels} is not a perfect square.")
    return side


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    - Calculates frequency of each class.
    - Inverts frequencies so rare classes get higher weights.
    - Normalizes so weights have mean 1.0.
    Output is a tensor of weights indexed by class id.
    """
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float64) / counts.sum()   # relative frequency per class
    inv = 1.0 / np.maximum(freq, 1e-12)              # inverse frequency
    inv = inv / inv.mean()                           # normalize mean to 1
    weights = np.ones(int(classes.max()) + 1, dtype=np.float64)
    weights[classes] = inv
    return torch.tensor(weights, dtype=torch.float32)


# =========================================
# Dataset class
# =========================================
class FERPixelCSV(Dataset):
    """
    PyTorch Dataset for Facial Emotion Recognition from CSV.

    CSV assumptions:
    ----------------
    - One column with labels (candidate names: 'emotion', 'label', 'class', 'target').
    - One column with pixel strings (candidate names: 'pixels', ' pixel', ' pixels', 'pixel_values').
    - Pixel values are space-separated grayscale intensities (0-255).
    - Each image is square (e.g. 48x48).

    Workflow:
    ---------
    - Load CSV via pandas.
    - Identify correct label and pixel columns (case/whitespace insensitive).
    - Store labels as integers.
    - On __getitem__, parse pixel string → NumPy array → PIL Image → RGB conversion (optional) → transforms.
    """

    LABEL_CANDS  = ("emotion", "label", "class", "target")
    PIXELS_CANDS = ("pixels", " pixel", " pixels", "pixel_values")

    def __init__(self, csv_path: str, transform=None, to_rgb: bool = True, source_side: Optional[int] = None):
        super().__init__()
        self.csv_path = csv_path
        self.transform = transform
        self.to_rgb = to_rgb

        # Try to robustly load CSV (detect separator automatically, handle BOM)
        try:
            self.df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            self.df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Normalize column names (lowercase, strip spaces) to match candidates
        norm_map = {c.lower().strip(): c for c in self.df.columns}
        self.label_col = self._find_col(norm_map, self.LABEL_CANDS)
        self.pixels_col = self._find_col(norm_map, self.PIXELS_CANDS)

        # If required columns are missing → raise error
        if self.label_col is None or self.pixels_col is None:
            raise KeyError(
                f"Could not find label/pixels columns in {csv_path}. "
                f"Found columns: {list(self.df.columns)}"
            )

        # Force labels to integers (handles stray characters by regex extracting numbers)
        self.df[self.label_col] = self.df[self.label_col].astype(str).str.extract(r"(-?\d+)").astype(int)
        self.labels = self.df[self.label_col].to_numpy()

        # If source_side not given, infer image side from first row’s pixel count
        if source_side is None:
            n = len(str(self.df[self.pixels_col].iloc[0]).split())
            self.source_side = _detect_square_side(n)
        else:
            self.source_side = int(source_side)

    @staticmethod
    def _find_col(norm_map: Dict[str, str], candidates: Tuple[str, ...]) -> Optional[str]:
        """
        Attempt to match actual CSV columns with candidate names.
        - First try exact lowercase matches.
        - Then try relaxed matches (remove spaces/dashes, allow suffix/prefix).
        """
        for cand in candidates:
            k = cand.lower().strip()
            if k in norm_map:
                return norm_map[k]
        for k_norm, orig in norm_map.items():
            kk = k_norm.replace(" ", "").replace("-", "_")
            for cand in candidates:
                cc = cand.lower().strip().replace(" ", "").replace("-", "_")
                if kk == cc or kk.endswith(cc) or cc.endswith(kk):
                    return orig
        return None

    def __len__(self) -> int:
        """Return dataset length (number of rows in CSV)."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Return one (image, label) pair.
        Steps:
        1. Extract pixel string.
        2. Convert to NumPy array of shape (side, side).
        3. Convert to PIL grayscale image.
        4. Optionally convert grayscale → RGB (to match pretrained CNN input).
        5. Apply transforms (augmentation/resize/normalize).
        """
        px_str = str(self.df[self.pixels_col].iloc[idx])
        arr = np.asarray(px_str.split(), dtype=np.uint8).reshape(self.source_side, self.source_side)

        img = Image.fromarray(arr, mode="L")  # grayscale image
        if self.to_rgb:
            img = img.convert("RGB")  # CNN backbones expect 3 channels

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.labels[idx])
        return img, label


# =========================================
# Transform builders
# =========================================
def build_transforms(image_size: int = 224):
    """
    Build torchvision transforms for training and evaluation.
    - Training includes data augmentation (random crop, flip, jitter).
    - Evaluation is deterministic (center crop, resize).
    Both normalize to ImageNet mean/std so pretrained models behave well.
    """
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


# =========================================
# DataLoader builder
# =========================================
def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
):
    """
    Build train/val/test DataLoaders.
    - Loads datasets via FERPixelCSV.
    - Applies transforms.
    - Optionally uses WeightedRandomSampler to mitigate class imbalance.
    Returns loaders + meta info dict.
    """
    train_tf, eval_tf = build_transforms(image_size=image_size)

    train_ds = FERPixelCSV(train_csv, transform=train_tf)
    val_ds   = FERPixelCSV(val_csv,   transform=eval_tf)
    test_ds  = FERPixelCSV(test_csv,  transform=eval_tf) if test_csv else None

    # Handle imbalance: WeightedRandomSampler gives higher probability to rare classes
    if use_weighted_sampler:
        class_weights = _compute_class_weights(train_ds.labels)    # weight per class
        sample_weights = class_weights[train_ds.labels]            # weight per sample
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # same as dataset size
            replacement=True                  # samples with replacement
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        # Simple random shuffle if sampler not used
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )

    # Validation loader (no shuffle)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Optional test loader
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    # Meta-information useful for training scripts
    meta = {
        "num_classes": len(np.unique(train_ds.labels)),
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds) if test_ds is not None else 0,
        "image_size": image_size,
        "sampler": "WeightedRandomSampler" if use_weighted_sampler else "Shuffle",
    }
    return train_loader, val_loader, test_loader, meta


# =========================================
# Smoke test
# =========================================
if __name__ == "__main__":
    """
    Running this file directly will attempt to load DataLoaders from 
    train.csv/val.csv/test.csv in the current folder and print a batch.
    Useful for quick debugging.
    """
    try:
        tr, va, te, meta = get_dataloaders(
            "train.csv", "val.csv", "test.csv",
            image_size=224, batch_size=32, num_workers=0  # num_workers=0 avoids Windows issues
        )
        xb, yb = next(iter(tr))
        print("OK - batch:", xb.shape, yb.shape, "unique labels:", yb.unique().tolist())
        print(meta)
    except Exception as e:
        print("Smoke test failed:", repr(e))
