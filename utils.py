import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import os

def get_dataloaders(data_dir='./PneumoniaMNIST', batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(),
        transforms.ToTensor()
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val)

    class_counts = np.bincount([label for _, label in train_set])
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for _, label in train_set]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    return model
