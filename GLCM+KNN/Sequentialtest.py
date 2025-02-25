import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define new dataset paths
train_path = r"C:\Users\User\Desktop\ivan files\Thesis\Dataset\PlantVillage\Corn (Maize)\Train"
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

# Load Training and Testing Datasets
X_train, y_train = load_dataset(train_path)  # Load training data
X_test, y_test = load_dataset(test_path)     # Load testing data

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)  # Use same encoder for test labels

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

# Extract GLCM features from training images
X_train_features = np.array([extract_glcm_features(img) for img in X_train])

# Train a simple KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_features, y_train_encoded)

# Function to segment leaf
def segment_leaf(image):
    if image is None or image.size == 0:
        return -1  # Return an invalid label for empty images

    features = np.array(extract_glcm_features(image)).reshape(1, -1)
    return knn.predict(features)[0]

# Apply segmentation to test images
segmented_labels = [segment_leaf(img) for img in X_test]

# Split dataset before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_train_features, y_train_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)  # Normalize test data too

print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_resampled))

# Build a simple classification model
model = Sequential([
    Flatten(input_shape=(5,)),  # 5 GLCM features
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='softmax')  # Binary classification
])

# Compile & train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Convert labels to categorical
y_encoded = label_encoder.fit_transform(y_train)

# Train the second model for disease classification
model2 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(set(y_encoded)), activation='softmax')  # Multi-class classification
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# def predict_plant_health(image):
#     features = np.array(extract_glcm_features(image)).reshape(1, -1)
    
#     # Step 1: Classify as Healthy or Diseased
#     health_pred = model.predict(features)
    
#     if health_pred < 0.5:
#         return "Healthy"
    
#     # Step 2: If diseased, classify the specific disease
#     disease_pred = model2.predict(features)
#     disease_label = label_encoder.inverse_transform([np.argmax(disease_pred)])[0]
    
#     return f"Diseased - {disease_label}"

def predict_plant_health(features):
    """ Predict plant health based on extracted features (not images). """

    features = np.array(features).reshape(1, -1)  # Ensure it's the correct shape

    # Step 1: Classify as Healthy or Diseased
    health_pred = model.predict(features)
    predicted_class = np.argmax(health_pred)  # 🔥 Get the class with the highest probability
    
    if predicted_class == 0:  # Assuming class 0 is "Healthy"
     return "Healthy"

    # Step 2: If diseased, classify the specific disease
    disease_label = label_encoder.inverse_transform([predicted_class])[0]
    return f"Diseased - {disease_label}"
    

# # Test with a sample image
# test_image = X_test[0]  # Selecting the first test image

# # Debugging: Print the shape of test_image
# print(f"Test image shape: {test_image.shape}")

# # Ensure the image is not None and has the correct shape
# if test_image is None or test_image.size == 0:
#     raise ValueError("Error: Test image is empty or None.")

# # If the image has an incorrect shape, convert it to grayscale
# if len(test_image.shape) == 3:  # If it's RGB, convert it
#     test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# # Ensure it's a 2D array
# if len(test_image.shape) != 2:
#     raise ValueError(f"Error: Test image has an invalid shape {test_image.shape}")

# # Proceed with prediction
# result = predict_plant_health(test_image)
# print("Prediction:", result)

test_features = scaler.transform([X_test[0]])  # Ensure it's normalized
result = predict_plant_health(test_features[0])  # Pass the first feature vector
print("Prediction:", result)
