import os
from dataset import get_dataloaders

# resolve file paths relative to this script's folder
BASE = os.path.dirname(os.path.abspath(__file__))
train = os.path.join(BASE, "train.csv")
val   = os.path.join(BASE, "val.csv")
test  = os.path.join(BASE, "test.csv")

# check that the files really exist
for p in [train, val, test]:
    print(p, "exists:", os.path.exists(p))

# build dataloaders
train_loader, val_loader, test_loader, meta = get_dataloaders(
    train, val, test,
    image_size=224, batch_size=32, num_workers=0  # num_workers=0 for Windows sanity test
)

print("META:", meta)

# get one batch
xb, yb = next(iter(train_loader))
print("Batch X shape:", xb.shape)   # expect [32, 3, 224, 224]
print("Batch y shape:", yb.shape)   # expect [32]
print("Unique labels in batch:", yb.unique())
