import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datetime import datetime

# Define dataset paths
train_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Combined\Train"
val_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Combined\Val"
test_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Combined\Test"

# Define image size for ResNet
IMG_SIZE = (224, 224)

# Function to load and preprocess images
def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not load image {img_path}")
        return None
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize pixel values
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor (C, H, W)
    return image

# Function to load dataset
def load_dataset(folder):
    images, labels = [], []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if not img_name.lower().endswith(valid_extensions):
                    continue

                image = load_image(img_path)
                if image is None:
                    continue

                images.append(image)
                labels.append(category)

    return images, labels

# Load datasets
X_train, y_train = load_dataset(train_path)
X_val, y_val = load_dataset(val_path)
X_test, y_test = load_dataset(test_path)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Convert to PyTorch tensors
X_train = torch.stack(X_train)
X_val = torch.stack(X_val)
X_test = torch.stack(X_test)
y_train = torch.tensor(y_train_encoded, dtype=torch.long)
y_val = torch.tensor(y_val_encoded, dtype=torch.long)
y_test = torch.tensor(y_test_encoded, dtype=torch.long)

# Define PyTorch dataset and dataloader
class PlantDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_loader = DataLoader(PlantDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(PlantDataset(X_val, y_val), batch_size=16, shuffle=False)
test_loader = DataLoader(PlantDataset(X_test, y_test), batch_size=16, shuffle=False)

# Load pre-trained ResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))  # Adjust final layer
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Track training progress
training_log = []

# Train Model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    training_log.append([epoch+1, epoch_loss, epoch_acc])
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Save training log as CSV
df_train_log = pd.DataFrame(training_log, columns=["Epoch", "Loss", "Accuracy"])
df_train_log["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_train_log.to_csv("training_log.csv", mode='a', index=False, header=not os.path.exists("training_log.csv"))

# Save trained model
torch.save(model.state_dict(), "resnet50_model.pth")
print("✅ Model saved as 'resnet50_model.pth'")

# Evaluate Model
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Save classification report
report_dict = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_report.to_csv("model_results.csv", mode='a', index=True, header=not os.path.exists("model_results.csv"))

# Save report to TXT
report_txt = f"""
🔹 Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ Accuracy: {accuracy:.4f}
✅ Precision: {precision:.4f}
✅ Recall: {recall:.4f}
✅ F1-score: {f1:.4f}

🔹 Per-Class Performance:
{classification_report(y_true, y_pred, target_names=label_encoder.classes_)}
"""
with open("model_results.txt", "a", encoding="utf-8") as file:
    file.write(report_txt + "\n" + "-"*50 + "\n")

print("✅ Results saved in 'model_results.csv' and 'model_results.txt'")
