import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Placeholder RGB leaf image tensor generation
        img = Image.new("RGB", (224, 224), color=(0, 255, 0))
        label = self.data.iloc[idx]["label_id"]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

def train_dl_model():
    print("Setting up Deep Learning Architecture (ResNet-18 Transfer Learning)...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PlantDiseaseDataset("data.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 plant disease categories
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training CNN for 2 Epochs...")
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/2 - Loss: {running_loss / len(dataloader):.4f}")
        
    print("\nTraining completed. Model ready for plant disease diagnostics.")

if __name__ == "__main__":
    train_dl_model()