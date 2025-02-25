import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load Dataset (Specify your dataset path)
dataset_path = r"C:\Users\User\Desktop\ivan files\Thesis\Sequential Approach\PlantVillage"
image_size = (224, 224)  # Resize images

def load_images_and_labels(dataset_path, split="train"):
    images = []
    labels = []
    for plant_type in os.listdir(dataset_path):
        plant_path = os.path.join(dataset_path, plant_type, split)
        if os.path.isdir(plant_path):
            for img_name in os.listdir(plant_path):
                img_path = os.path.join(plant_path, img_name)
                img = load_img(img_path, target_size=image_size)
                img = img_to_array(img)
                img = preprocess_input(img)
                images.append(img)
                labels.append(plant_type)
    return np.array(images), np.array(labels)

# Load training, validation, and test sets
X_train, y_train = load_images_and_labels(dataset_path, "train")
X_val, y_val = load_images_and_labels(dataset_path, "val")
X_test, y_test = load_images_and_labels(dataset_path, "test")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Apply SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train_encoded)
X_train_resampled = X_train_resampled.reshape(-1, *image_size, 3)

# Function to Extract GLCM Features
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

# Extract Features for KNN Training
glcm_features = np.array([extract_glcm_features(img) for img in X_train_resampled])
scaler = StandardScaler()
X_train_glcm_scaled = scaler.fit_transform(glcm_features)

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_glcm_scaled, y_train_resampled)

# Function to Predict using Sequential Approach
def predict_leaf_disease(image):
    glcm_features = np.array(extract_glcm_features(image)).reshape(1, -1)
    glcm_features_scaled = scaler.transform(glcm_features)
    
    prediction = knn.predict(glcm_features_scaled)[0]
    
    if label_encoder.inverse_transform([prediction])[0] == 'Healthy':
        return 'Healthy'
    else:
        # Use ResNet50V2 for Disease Classification
        model = ResNet50V2(weights='imagenet', include_top=True)
        img_resized = cv2.resize(image, image_size)
        img_preprocessed = preprocess_input(img_resized)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        resnet_prediction = model.predict(img_expanded)
        class_index = np.argmax(resnet_prediction)
        class_label = label_encoder.inverse_transform([class_index])[0]
        return class_label

# Example Prediction
test_img = X_test[0]
result = predict_leaf_disease(test_img)
print("Prediction:", result)
