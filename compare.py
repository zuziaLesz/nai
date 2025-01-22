import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from lab import PlantDiseaseDataset, transform

test_dataset = PlantDiseaseDataset(csv_file='test_labels.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save the DataLoader to a file (you cannot save the DataLoader directly)
torch.save({
    'test_dataset': test_dataset,
    'transform': transform
},'test_loader.path')
# Function to evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds

# Load individual models and dataloaders (replace with your actual model paths and dataloaders)
cnn_model = torch.load('CNN.pth')
resnet_model = torch.load('Resnet.pth')
rainforest_model = torch.load('RainForest.pth')

test_loader = torch.load('test_loader.pth')  # Assuming the same test_loader is used for all models

# Evaluate each model
cnn_labels, cnn_preds = evaluate_model(cnn_model, test_loader)
resnet_labels, resnet_preds = evaluate_model(resnet_model, test_loader)
rainforest_labels, rainforest_preds = evaluate_model(rainforest_model, test_loader)

# Calculate metrics
cnn_precision = precision_score(cnn_labels, cnn_preds, average='binary')
cnn_recall = recall_score(cnn_labels, cnn_preds, average='binary')
cnn_f1 = f1_score(cnn_labels, cnn_preds, average='binary')

resnet_precision = precision_score(resnet_labels, resnet_preds, average='binary')
resnet_recall = recall_score(resnet_labels, resnet_preds, average='binary')
resnet_f1 = f1_score(resnet_labels, resnet_preds, average='binary')

rainforest_precision = precision_score(rainforest_labels, rainforest_preds, average='binary')
rainforest_recall = recall_score(rainforest_labels, rainforest_preds, average='binary')
rainforest_f1 = f1_score(rainforest_labels, rainforest_preds, average='binary')

# Metrics
models = ["CNN", "ResNet", "RainForest"]
precision = [cnn_precision, resnet_precision, rainforest_precision]
recall = [cnn_recall, resnet_recall, rainforest_recall]
f1_score_values = [cnn_f1, resnet_f1, rainforest_f1]

# Define positions for the bars
x = np.arange(len(models))  # Label locations
width = 0.25  # Width of the bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score_values, width, label='F1-Score')

# Add some text for labels, title and axes
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('Comparison of Model Performance Metrics', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend()

# Attach a text label above each bar, displaying its height
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text slightly above the bar
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# Display the plot
plt.tight_layout()
plt.show()