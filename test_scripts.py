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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Define dataset paths
train_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Train"
val_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Validation"
test_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Test"

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

# Encode labels (Healthy = 0, Diseased = 1)
label_encoder = LabelEncoder()
y_train_encoded = (label_encoder.fit_transform(y_train) > 0).astype(int)  # Convert to binary labels
y_val_encoded = (label_encoder.transform(y_val) > 0).astype(int)
y_test_encoded = (label_encoder.transform(y_test) > 0).astype(int)

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
model1 = models.resnet50(weights="IMAGENET1K_V2")
model1.fc = nn.Linear(model1.fc.in_features, 2)  # Modify final layer for binary classification
model1 = model1.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.0001)

# Train Model 1
epochs = 10
for epoch in range(epochs):
    model1.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model1(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model1.state_dict(), "healthy_vs_diseased_resnet.pth")
print("✅ Model saved as 'healthy_vs_diseased_resnet.pth'")

# Function to predict Healthy vs. Diseased
def predict_health(image):
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    model1.eval()
    with torch.no_grad():
        output = model1(image)
    predicted_class = torch.argmax(output).item()
    return "Healthy" if predicted_class == 0 else "Diseased"

# Evaluate Model 1 on test data
y_true, y_pred = [], []
model1.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model1(images)
        predictions = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Compute per-class performance
report = classification_report(y_true, y_pred, target_names=['Healthy', 'Diseased'])

# Save evaluation results
with open("healthy_vs_diseased_pytorch_results.txt", "w", encoding="utf-8") as file:
    file.write(f"🔹 Overall Model Performance:\n")
    file.write(f"✅ Accuracy: {accuracy:.4f}\n")
    file.write(f"✅ Precision (weighted avg): {precision:.4f}\n")
    file.write(f"✅ Recall (weighted avg): {recall:.4f}\n")
    file.write(f"✅ F1-score (weighted avg): {f1:.4f}\n\n")
    file.write("🔹 Per-Class Performance:\n")
    file.write(report)

# Print results to console
print("\n🔹 Overall Model Performance:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision (weighted avg): {precision:.4f}")
print(f"✅ Recall (weighted avg): {recall:.4f}")
print(f"✅ F1-score (weighted avg): {f1:.4f}")

print("\n🔹 Per-Class Performance:")
print(report)

# Test a sample prediction
sample_image = X_test[0].to(device)
result = predict_health(sample_image)
print("\n🔹 Sample Prediction:", result)
