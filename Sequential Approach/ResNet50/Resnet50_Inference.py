import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import joblib

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  # Initialize ResNet
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
model.load_state_dict(torch.load("healthy_vs_diseased_resnet.pth", map_location=device))
model = model.to(device)
model.eval()  # Set to evaluation mode

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Define image transformation (ResNet50 requires normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load new dataset
def load_new_test_data(folder):
    images, labels = [], []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if not img_name.lower().endswith(valid_extensions):
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue

                image = transform(image)  # Apply transformation
                images.append(image)
                labels.append(category)

    return torch.stack(images), labels

# Path to the new test dataset
new_test_path = r"C:\Users\User\Desktop\NewTestDataset"

# Load new test dataset
X_new_test, y_new_test = load_new_test_data(new_test_path)

# Predict labels
y_pred_new = []
with torch.no_grad():
    for img in X_new_test:
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        output = model(img)
        predicted_class = torch.argmax(output).item()
        y_pred_new.append(label_encoder.inverse_transform([predicted_class])[0])

# Print results
for img_name, prediction in zip(os.listdir(new_test_path), y_pred_new):
    print(f"Image: {img_name} -> Prediction: {prediction}")
