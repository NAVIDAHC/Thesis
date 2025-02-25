import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Define dataset paths
train_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Train"
val_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Val"
test_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Test"

# Function to check if an image is valid
def load_image(img_path):
    """Try loading an image with OpenCV, fallback to PIL if needed."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:  # If OpenCV fails, use PIL
        try:
            pil_image = Image.open(img_path).convert("L")  # Convert to grayscale
            image = np.array(pil_image)
        except Exception as e:
            print(f"Warning: PIL also failed to load image {img_path} -> {e}")
            return None
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
                    print(f"Skipping non-image file: {img_path}")
                    continue

                image = load_image(img_path)
                if image is None:
                    print(f"Warning: Skipping unreadable image -> {img_path}")
                    continue

                image = cv2.resize(image, (256, 256))
                images.append(image)
                labels.append(category)

    return np.array(images), np.array(labels)

# Load datasets
X_train, y_train = load_dataset(train_path)  # Training data
X_val, y_val = load_dataset(val_path)        # Validation data
X_test, y_test = load_dataset(test_path)     # Testing data

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Feature extraction function using GLCM
def extract_glcm_features(image):
    if image is None or image.size == 0:
        return np.zeros(5)  # Return zero features if the image is empty

    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, dissimilarity, homogeneity, energy, correlation]

# Extract GLCM features
X_train_features = np.array([extract_glcm_features(img) for img in X_train])
X_val_features = np.array([extract_glcm_features(img) for img in X_val])
X_test_features = np.array([extract_glcm_features(img) for img in X_test])

# Apply SMOTE only to training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train_encoded)

# Normalize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_val_features = scaler.transform(X_val_features)  # Normalize validation data
X_test_features = scaler.transform(X_test_features)  # Normalize test data

print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_resampled))

# Build a classification model
model = Sequential([
    Flatten(input_shape=(5,)),  # 5 GLCM features
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(set(y_train_encoded)), activation='softmax')  # Multi-class classification
])

# Compile & train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_resampled, y_train_resampled, epochs=30, batch_size=8, validation_data=(X_val_features, y_val_encoded))

# Function to predict plant health
def predict_plant_health(features):
    """ Predict plant health based on extracted features. """
    features = np.array(features).reshape(1, -1)  # Ensure correct shape
    health_pred = model.predict(features)
    predicted_class = np.argmax(health_pred)  # Get class with highest probability
    return label_encoder.inverse_transform([predicted_class])[0]

# Evaluate on test data
y_pred = model.predict(X_test_features)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Compute evaluation metrics
accuracy = accuracy_score(y_test_encoded, y_pred_classes)
precision = precision_score(y_test_encoded, y_pred_classes, average='weighted')
recall = recall_score(y_test_encoded, y_pred_classes, average='weighted')
f1 = f1_score(y_test_encoded, y_pred_classes, average='weighted')

# Print overall results
print("\n🔹 Overall Model Evaluation on Test Data:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision (weighted avg): {precision:.4f}")
print(f"✅ Recall (weighted avg): {recall:.4f}")
print(f"✅ F1-score (weighted avg): {f1:.4f}")

# Print classification report for per-class metrics
print("\n🔹 Per-Class Performance:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))

# Save evaluation results to a text file
results_text = f"""
🔹 Disease Classification Evaluation:
✅ Accuracy: {accuracy:.4f}
✅ Precision: {precision:.4f}
✅ Recall: {recall:.4f}
✅ F1-score: {f1:.4f}

🔹 Per-Class Performance:
{classification_report(y_test_encoded[y_test_encoded > 0], y_pred_disease, target_names=label_encoder.classes_[1:])}
"""

# Write to a file
with open("model_results.txt", "w") as file:
    file.write(results_text)

print("✅ Model evaluation results saved in 'model_results.txt'")


# Test with a sample image
test_features = scaler.transform([X_test_features[0]])  # Normalize
result = predict_plant_health(test_features[0])  # Get prediction
print("\n🔹 Sample Prediction:", result)

# Save the first model (Healthy vs. Diseased)
model.save("plant_health_model.h5")
print("✅ First model saved as 'plant_health_model.h5'")

