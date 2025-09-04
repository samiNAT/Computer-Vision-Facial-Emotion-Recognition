#!/usr/bin/env python3
"""
Robust splitter for FER-like CSVs into stratified train/val/test.
- Expects one row per image.
- Label column can be named: emotion | label | class | target (case-insensitive)
- Pixels column can be named: pixels | pixel | pixel_values (case-insensitive)
- If a 'Usage' column exists (FER2013 style), it's ignored for the 70/15/15 split unless you adapt below.

Outputs (in same folder as script):
  - train.csv, val.csv, test.csv
  - data_with_split.csv
  - classes.json
Also prints a summary.
"""

import os
import json
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------- config -------------
SPLIT_SEED = 42
TRAIN_PCT = 0.70
VAL_PCT = 0.15
TEST_PCT = 0.15
# ----------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "data.csv")
TRAIN = os.path.join(BASE, "train.csv")
VAL   = os.path.join(BASE, "val.csv")
TEST  = os.path.join(BASE, "test.csv")
WITH_SPLIT = os.path.join(BASE, "data_with_split.csv")
CLASSES = os.path.join(BASE, "classes.json")

LABEL_ALIASES = {"emotion", "label", "class", "target"}
PIXEL_ALIASES = {"pixels", "pixel", "pixel_values"}

def find_column(df, aliases, friendly_name):
    # Create a mapping from normalized -> original
    norm_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in norm_map:
            return norm_map[alias]
    # try small variants like trailing 's' or underscores
    for c_norm, c_orig in norm_map.items():
        name = c_norm.replace(" ", "").replace("-", "_")
        for alias in aliases:
            if alias in name or name in alias:
                return c_orig
    return None

def read_csv_safely(path):
    # Try with sep=None (auto-detect) and UTF-8 with BOM handling
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        # Fallback to default comma
        return pd.read_csv(path, encoding="utf-8-sig")

def main():
    assert os.path.isfile(SRC), f"CSV not found next to script: {SRC}"

    df = read_csv_safely(SRC)

    # Normalize column detection (case-insensitive, trims)
    label_col = find_column(df, LABEL_ALIASES, "label")
    pixels_col = find_column(df, PIXEL_ALIASES, "pixels")

    if label_col is None or pixels_col is None:
        print("\n[!] Could not automatically detect the required columns.")
        print("    Available columns:", list(df.columns))
        print("    Expected label column to be one of (case-insensitive):", sorted(LABEL_ALIASES))
        print("    Expected pixels column to be one of (case-insensitive):", sorted(PIXEL_ALIASES))
        raise AssertionError("CSV must contain a label column and a pixels column.")

    # Basic validations
    if df[label_col].isna().any():
        raise ValueError(f"Found NaNs in label column '{label_col}'.")
    if df[pixels_col].isna().any():
        raise ValueError(f"Found NaNs in pixels column '{pixels_col}'.")

    # Coerce label to int
    try:
        df[label_col] = df[label_col].astype(int)
    except Exception:
        # try to map strings like "0","1" etc.
        df[label_col] = df[label_col].astype(str).str.extract(r"(\d+)").astype(int)

    # --- Stratified split: 70/15/15 ---
    df_train, df_tmp = train_test_split(
        df, test_size=(1.0 - TRAIN_PCT), stratify=df[label_col],
        random_state=SPLIT_SEED, shuffle=True
    )
    # Split remaining equally for val/test
    val_ratio_of_tmp = VAL_PCT / (VAL_PCT + TEST_PCT)  # typically 0.5
    df_val, df_test = train_test_split(
        df_tmp, test_size=(1.0 - val_ratio_of_tmp), stratify=df_tmp[label_col],
        random_state=SPLIT_SEED, shuffle=True
    )

    # Save separate CSVs
    df_train.to_csv(TRAIN, index=False)
    df_val.to_csv(VAL, index=False)
    df_test.to_csv(TEST, index=False)

    # Combined CSV with split column
    dft  = df_train.copy(); dft["split"] = "train"
    dfv  = df_val.copy();   dfv["split"] = "val"
    dfts = df_test.copy();  dfts["split"] = "test"
    df_all = pd.concat([dft, dfv, dfts], axis=0).reset_index(drop=True)
    df_all.to_csv(WITH_SPLIT, index=False)

    # classes.json
    classes_sorted = sorted(pd.unique(df[label_col]).tolist())
    with open(CLASSES, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes_sorted,
            "class_to_index": {str(c): int(c) for c in classes_sorted}
        }, f, indent=2)

    # Print a quick summary
    def dist(d): return dict(Counter(d[label_col].tolist()))
    summary = {
        "csv_path": SRC,
        "columns_used": {"label": label_col, "pixels": pixels_col},
        "total": len(df),
        "splits": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "class_dist_total": dist(df),
        "class_dist_train": dist(df_train),
        "class_dist_val": dist(df_val),
        "class_dist_test": dist(df_test),
        "outputs": {
            "train": TRAIN, "val": VAL, "test": TEST,
            "data_with_split": WITH_SPLIT, "classes_json": CLASSES
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
