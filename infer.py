import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from dataset import FERPixelCSV, build_transforms
from model import get_resnet18
from utils import get_device, load_checkpoint

# ---- Paths ----
BASE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE, "test.csv")
CKPT_PATH = os.path.join(BASE, "outputs", "best_model.pth")
OUT_DIR = os.path.join(BASE, "outputs")

# ---- Human-readable labels ----
IDX_TO_CLASS = {
    0: "anger",
    1: "disgust",
    2: "sad",
    3: "happiness",
    4: "surprise",
}


def draw_overlay(pil_img: Image.Image, text: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font_size = max(16, img.height // 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    pad = 6
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    rect = [10 - pad, 10 - pad, 10 + tw + pad, 10 + th + pad]
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return img


def predict_tensor(model, device, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs) * 100.0)
    return pred, conf, probs


# ===============================
# Reusable API function
# ===============================
def predict_from_image(image_path: str) -> str:
    """
    Run inference on a .jpg/.png image and return the predicted emotion as string.
    Also plots the image with overlay text.
    """
    device = get_device()
    _, eval_tf = build_transforms(image_size=224)

    # Prepare model
    test_ds_proc = FERPixelCSV(TEST_CSV, transform=eval_tf)
    num_classes = len(np.unique(test_ds_proc.labels))
    model = get_resnet18(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, CKPT_PATH, device)

    # Load and preprocess image
    pil_img = Image.open(image_path).convert("RGB")
    x = eval_tf(pil_img).unsqueeze(0)

    # Prediction
    pred, conf, _ = predict_tensor(model, device, x)
    pred_name = IDX_TO_CLASS.get(pred, str(pred))

    # Show image with overlay
    overlay = draw_overlay(pil_img, f"{pred_name} ({conf:.0f}%)")
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Prediction: {pred_name} ({conf:.0f}%)")
    plt.show()

    return pred_name


# ===============================
# CLI main function
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Pick a row from test.csv")
    parser.add_argument("--image", type=str, default=None, help="Path to external image (JPG/PNG)")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    _, eval_tf = build_transforms(image_size=224)
    test_ds_proc = FERPixelCSV(TEST_CSV, transform=eval_tf)
    test_ds_raw = FERPixelCSV(TEST_CSV, transform=None)
    num_classes = len(np.unique(test_ds_proc.labels))

    model = get_resnet18(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, CKPT_PATH, device)

    # Option A: external image
    if args.image:
        emotion = predict_from_image(args.image)
        print(f"Prediction: {emotion}")
        return

    # Option B: dataset index
    if args.index is not None:
        pil_img, true_label = test_ds_raw[args.index]
        img_proc, _ = test_ds_proc[args.index]
        x = img_proc.unsqueeze(0)
        pred, conf, _ = predict_tensor(model, device, x)
        pred_name = IDX_TO_CLASS.get(pred, str(pred))
        true_name = IDX_TO_CLASS.get(int(true_label), str(true_label))

        print(f"\nPrediction for test.csv index {args.index}:")
        print(f"- true     : {true_label} ({true_name})")
        print(f"- predicted: {pred} ({pred_name}) conf={conf:.1f}%")

        overlay = draw_overlay(pil_img.resize((224, 224)),
                               f"True: {true_name} | Pred: {pred_name} ({conf:.0f}%)")
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"True: {true_name} | Pred: {pred_name} ({conf:.0f}%)")
        plt.show()
        return

    print("Nothing to do. Provide --image <path> or --index <int>.")


if __name__ == "__main__":
    main()


def predict_from_image(image_path: str) -> str:
    """
    Third option (callable from other files):
    Given a path to a JPG/PNG image, this will:
      1) Load the trained model
      2) Preprocess the image
      3) Plot the ORIGINAL image with an overlay of the prediction
      4) Return the predicted emotion as a string

    Usage from another file:
        from infer import predict_from_image
        label = predict_from_image("/path/to/image.jpg")
        print(label)
    """
    # Device and transforms (same as CLI flow)
    device = get_device()
    _, eval_tf = build_transforms(image_size=224)

    # Determine number of classes from test.csv (same approach as main)
    test_ds_proc = FERPixelCSV(TEST_CSV, transform=eval_tf)
    num_classes = len(np.unique(test_ds_proc.labels))

    # Load model & checkpoint
    model = get_resnet18(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, CKPT_PATH, device)

    # Load image and run inference
    pil_img = Image.open(image_path).convert("RGB")
    x = eval_tf(pil_img).unsqueeze(0)
    pred, conf, _ = predict_tensor(model, device, x)
    pred_name = IDX_TO_CLASS.get(pred, str(pred))

    # Plot original image with overlayed prediction (unchanged style)
    overlay = draw_overlay(pil_img, f"{pred_name} ({conf:.0f}%)")
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Prediction: {pred_name} ({conf:.0f}%)")
    plt.show()

    return pred_name
