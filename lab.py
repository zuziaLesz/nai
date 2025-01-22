import kagglehub
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim


class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]
        img = Image.open(img_path).convert("RGB")

        # Convert label to integer (1 = healthy, 0 = not_healthy)
        label = 1 if label == 'healthy' else 0

        if self.transform:
            img = self.transform(img)

        return img, label

#Define your image transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size expected by most models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# # Load your datasets
train_dataset = PlantDiseaseDataset(csv_file='train_labels.csv', transform=transform)
valid_dataset = PlantDiseaseDataset(csv_file='valid_labels.csv', transform=transform)
#
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
#
# Load a pre-trained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes (healthy, not_healthy)

# Move model to the GPU if available (recommended for faster training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print training stats
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation loop (optional)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
model_path = "Resnet.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
