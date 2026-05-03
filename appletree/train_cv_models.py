"""
========================================================================
TRAIN CV-BASED MODELS (KEEPS ORIGINAL .PKL FILES SAFE)
========================================================================
Saves all NEW models with 'cv_' prefix - your existing models stay safe.

OUTPUT FILES (in appletree folder):
  - cv_apple_disease_model.pkl
  - cv_scaler.pkl
  - cv_pca.pkl
  - cv_label_encoder.pkl
  - cv_anomaly_detector.pkl

Original files UNTOUCHED:
  - apple_disease_model.pkl, scaler.pkl, pca.pkl,
    label_encoder.pkl, anomaly_detector.pkl
========================================================================
"""

import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'train')
IMG_SIZE = (64, 64)


def extract_cv_features(image):
    """
    Extract Computer Vision features using OpenCV.
    
    Features:
    1. HSV color histograms (24 bins x 3 channels = 72 features)
    2. Canny edge density (1 feature)
    3. Channel statistics for BGR + HSV (12 features)
    4. Laplacian variance for texture (1 feature)
    Total: 86 meaningful features
    
    !!! THIS FUNCTION MUST BE IDENTICAL IN apple.py !!!
    """
    features = []
    image = cv2.resize(image, IMG_SIZE)
    
    # 1. HSV Color Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [24], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # 2. Canny Edge Density
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # 3. Channel Statistics (BGR + HSV)
    for channel in range(3):
        features.append(np.mean(image[:, :, channel]))
        features.append(np.std(image[:, :, channel]))
    for channel in range(3):
        features.append(np.mean(hsv[:, :, channel]))
        features.append(np.std(hsv[:, :, channel]))
    
    # 4. Texture (Laplacian Variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.var())
    
    return np.array(features)


# ============= LOAD DATASET =============
print("=" * 70)
print("LOADING DATASET WITH CV FEATURES...")
print("=" * 70)

X_features = []
labels = []
class_folders = sorted(os.listdir(dataset_path))
print(f"Found classes: {class_folders}")

for class_name in class_folders:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    print(f"\nProcessing {class_name} ({len(images)} images)...")
    
    for img_name in tqdm(images, desc=class_name):
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_cv_features(img)
            X_features.append(features)
            labels.append(class_name)
        except Exception as e:
            print(f"Error with {img_path}: {e}")

X = np.array(X_features)
y = np.array(labels)
print(f"\n✅ Loaded {len(y)} images, {X.shape[1]} features each (vs 3072 in raw pixel approach)")


# ============= ENCODE & SCALE =============
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClasses: {list(label_encoder.classes_)}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# ============= PCA =============
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} components (95% variance)")


# ============= TRAIN =============
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTraining SVM on CV features...")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ CV-based SVM Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# ============= ANOMALY DETECTOR =============
print("Training Isolation Forest...")
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
anomaly_detector.fit(X_pca)


# ============= SAVE WITH cv_ PREFIX =============
print("\n" + "=" * 70)
print("SAVING CV MODELS (with cv_ prefix - originals stay safe)")
print("=" * 70)

model_files = {
    'cv_apple_disease_model.pkl': model,
    'cv_scaler.pkl':              scaler,
    'cv_pca.pkl':                 pca,
    'cv_label_encoder.pkl':       label_encoder,
    'cv_anomaly_detector.pkl':    anomaly_detector,
}

for filename, obj in model_files.items():
    filepath = os.path.join(BASE_DIR, filename)
    joblib.dump(obj, filepath)
    print(f"   ✅ Saved: {filename}")

print("\n" + "=" * 70)
print(f"🎉 SUCCESS! CV models saved.")
print(f"📁 Location: {BASE_DIR}")
print(f"📁 Original .pkl files are UNTOUCHED.")
print("=" * 70)
print("\nNext: Update apple.py to add the version toggle.")