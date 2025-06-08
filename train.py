# Training script for ResNet Pneumonia Classifier
from utils import get_dataloaders, build_model
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
train_loader, _ = get_dataloaders()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(20):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader.dataset):.4f}")
torch.save(model.state_dict(), "resnet_pneumonia.pth")
