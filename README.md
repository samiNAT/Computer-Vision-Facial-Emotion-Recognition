# 🎭 Face Emotion Detection  

Face Emotion Detection using deep learning (PyTorch).  
This project trains a **ResNet18-based CNN** to classify face expressions into:  
➡️ anger, disgust, sadness, happiness, and surprise.  

It includes **data preprocessing, training, evaluation, inference, and visualization tools** and is ready to run both **locally** and on **Google Colab**.  

---

## 📸 Example Predictions  

Here are some sample outputs from the trained model:  

<img width="542" height="613" alt="image" src="https://github.com/user-attachments/assets/0cc02d18-c2b6-456b-afd0-377f2c7c86a2" />

---

## ✨ Features  
- Stratified dataset splitting (`split_data.py`)  
- ResNet18 backbone with fine-tuned classifier head  
- Training with class balancing (WeightedRandomSampler)  
- Evaluation with accuracy, loss, and confusion matrix  
- Inference on test set or external images  
- Training history plots (accuracy & loss curves)  
- Fully Colab-ready  

---

## 📂 Dataset  

The project works with **FER-style CSV datasets**, where pixel values are stored as space-separated strings.  

- `split_data.py` automatically generates:  
  - `train.csv`, `val.csv`, `test.csv`  
  - `data_with_split.csv`  
  - `classes.json`

## Clone repository
git clone https://github.com/samiNAT/Face_Emotion_Detection.git
cd Face_Emotion_Detection

## Unzip datasets (adjust if needed)
unzip data.zip -d ./data
unzip train.zip -d .
unzip test.zip -d .
unzip val.zip -d .

## 🚀 Quick Start (Local)
1. Train
python train.py

2. Evaluate
python eval.py

3. Inference (external image)
python infer.py --image /path/to/image.jpg


Or from test set:
python infer.py --index 3

## 📊 Outputs
Training automatically saves results in ./outputs:
#### ✅ Best model checkpoint → best_model.pth
#### ✅ Training history → history.pkl
#### ✅ Accuracy curve → acc_curve.png
#### ✅ Loss curve → loss_curve.png

## ☁️ Run on Google Colab
The project runs smoothly on Google Colab with GPU. Example workflow:

import torch, platform
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

### 1. Mount Google Drive & set project folder
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/Emotional_Face"
!mkdir -p "$PROJECT_DIR"
%cd "$PROJECT_DIR"
!pwd

### 2. Install dependencies
!pip -q install scikit-learn matplotlib

### 3. Train model
!python train.py

### 4. Evaluate model
!python eval.py

### 5. Show training curves
from IPython.display import Image, display
display(Image(filename="outputs/acc_curve.png"))
display(Image(filename="outputs/loss_curve.png"))

### 6. Inference
%run infer.py --index 97   # you can select any index from test.csv
%run infer.py --image "/content/drive/MyDrive/Emotional_Face/sad.jpg"

## 📈 Results
Confusion Matrix for test set

## 📦 Requirements
Python 3.10+
torch, torchvision
numpy, pandas
scikit-learn, matplotlib, Pillow

Install via:
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Open a PR or issue to discuss improvements.

