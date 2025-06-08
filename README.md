# ResNet Pneumonia Classifier

This project fine-tunes a ResNet-50 model to classify chest X-rays as either pneumonia or normal using the PneumoniaMNIST dataset.

## 📦 Dataset
- **Source**: [PneumoniaMNIST on Kaggle](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

## 🛠️ Requirements
See `requirements.txt` for dependencies.

## 🧠 Model
- ResNet-50 pretrained on ImageNet
- Final layer modified for binary classification

## 📊 Evaluation
- Metrics: Recall, F1-Score, AUC-ROC

