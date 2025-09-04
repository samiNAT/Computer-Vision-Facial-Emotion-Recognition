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

Example label mapping (from `classes.json`):  

```json
{
  "classes": [0, 1, 2, 3, 4],
  "class_to_index": {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
  }
}

Installation
# Clone repository
git clone https://github.com/samiNAT/Face_Emotion_Detection.git
cd Face_Emotion_Detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

