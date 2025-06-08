# Evaluation script
from utils import get_dataloaders, build_model
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(torch.load("resnet_pneumonia.pth"))
model.eval()

_, val_loader = get_dataloaders()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:\n", classification_report(all_labels, all_preds))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("AUC-ROC:", roc_auc_score(all_labels, all_preds))
