"""
========================================================================
COMPUTER VISION vs CLASSICAL MACHINE LEARNING COMPARISON
========================================================================
This script compares two approaches for Apple Leaf Disease Detection:

APPROACH 1: COMPUTER VISION (OpenCV-based feature extraction + ML)
   - Uses OpenCV to extract meaningful features:
     * HSV Color Histograms
     * Canny Edge features
     * Statistical features (mean, std)
     * Texture features (Laplacian variance)

APPROACH 2: CLASSICAL ML (Raw Pixels + ML)
   - Resize image to 32x32
   - Flatten pixels (3072 features)
   - Feed directly to ML model

Both approaches use the same models (Random Forest, SVM, KNN)
for a fair comparison.

OUTPUT:
   - cv_vs_ml_results.csv  (numerical comparison)
   - cv_vs_ml_chart.png    (visual comparison)
   - approach_examples.png (visual examples of features)
========================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'train')
output_dir = os.path.join(BASE_DIR, 'output')
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = (64, 64)  # Slightly bigger for better CV features


# ===================== APPROACH 1: COMPUTER VISION FEATURES =====================
def extract_cv_features(image):
    """
    Extract Computer Vision features using OpenCV:
    1. HSV color histograms (24 bins x 3 channels = 72 features)
    2. Canny edge density (1 feature)
    3. Channel statistics (mean & std for B, G, R, H, S, V = 12 features)
    4. Laplacian variance for texture (1 feature)
    
    Total: ~86 meaningful features (instead of 12,288 raw pixels)
    """
    features = []
    
    # Resize for consistency
    image = cv2.resize(image, IMG_SIZE)
    
    # ---- 1. HSV Color Histogram ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [24], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # ---- 2. Canny Edge Density ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # ---- 3. Channel Statistics (BGR + HSV) ----
    for channel in range(3):
        features.append(np.mean(image[:, :, channel]))
        features.append(np.std(image[:, :, channel]))
    for channel in range(3):
        features.append(np.mean(hsv[:, :, channel]))
        features.append(np.std(hsv[:, :, channel]))
    
    # ---- 4. Texture Feature (Laplacian Variance) ----
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.var())
    
    return np.array(features)


# ===================== APPROACH 2: CLASSICAL ML FEATURES =====================
def extract_classical_features(image):
    """
    Classical ML approach: Just resize and flatten raw pixels.
    This is what the original project does.
    
    Output: 32 * 32 * 3 = 3072 features (raw pixel values)
    """
    image = cv2.resize(image, (32, 32))
    return image.flatten()


# ===================== LOAD DATASET =====================
print("=" * 70)
print("LOADING DATASET...")
print("=" * 70)

cv_features_list = []
classical_features_list = []
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
            
            # Extract features using BOTH approaches
            cv_feat = extract_cv_features(img)
            classical_feat = extract_classical_features(img)
            
            cv_features_list.append(cv_feat)
            classical_features_list.append(classical_feat)
            labels.append(class_name)
        except Exception as e:
            print(f"Error with {img_path}: {e}")

# Convert to numpy arrays
X_cv = np.array(cv_features_list)
X_classical = np.array(classical_features_list)
y = np.array(labels)

print(f"\n✅ Dataset loaded:")
print(f"   Total images: {len(y)}")
print(f"   CV features shape:        {X_cv.shape}  (per image: {X_cv.shape[1]} features)")
print(f"   Classical features shape: {X_classical.shape}  (per image: {X_classical.shape[1]} features)")


# ===================== ENCODE LABELS =====================
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ===================== SCALE FEATURES =====================
scaler_cv = MinMaxScaler()
X_cv_scaled = scaler_cv.fit_transform(X_cv)

scaler_classical = MinMaxScaler()
X_classical_scaled = scaler_classical.fit_transform(X_classical)


# ===================== TRAIN/TEST SPLIT =====================
X_cv_train, X_cv_test, y_train, y_test = train_test_split(
    X_cv_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_cl_train, X_cl_test, _, _ = train_test_split(
    X_classical_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# ===================== TRAIN AND EVALUATE =====================
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Train a model and return all metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }


print("\n" + "=" * 70)
print("TRAINING MODELS — APPROACH 1: COMPUTER VISION (OpenCV Features)")
print("=" * 70)

cv_results = []
models_cv = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', random_state=42),
    'KNN':           KNeighborsClassifier(n_neighbors=5),
}

for name, model in models_cv.items():
    print(f"  Training {name}...")
    res = evaluate_model(model, X_cv_train, X_cv_test, y_train, y_test, name)
    res['Approach'] = 'Computer Vision (OpenCV)'
    cv_results.append(res)
    print(f"    Accuracy: {res['Accuracy']:.4f}")


print("\n" + "=" * 70)
print("TRAINING MODELS — APPROACH 2: CLASSICAL ML (Raw Pixels)")
print("=" * 70)

classical_results = []
models_classical = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', random_state=42),
    'KNN':           KNeighborsClassifier(n_neighbors=5),
}

for name, model in models_classical.items():
    print(f"  Training {name}...")
    res = evaluate_model(model, X_cl_train, X_cl_test, y_train, y_test, name)
    res['Approach'] = 'Classical ML (Raw Pixels)'
    classical_results.append(res)
    print(f"    Accuracy: {res['Accuracy']:.4f}")


# ===================== SAVE RESULTS =====================
all_results = cv_results + classical_results
df = pd.DataFrame(all_results)
df = df[['Approach', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

csv_path = os.path.join(output_dir, 'cv_vs_ml_results.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved CSV: {csv_path}")

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)
print(df.to_string(index=False))


# ===================== VISUALIZATION 1: BAR CHART =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

# Plot 1: Computer Vision approach
ax1 = axes[0]
for i, model_name in enumerate(['Random Forest', 'SVM', 'KNN']):
    values = [r for r in cv_results if r['Model'] == model_name][0]
    scores = [values[m] for m in metrics]
    ax1.bar(x + i * width, scores, width, label=model_name)
ax1.set_title('APPROACH 1: Computer Vision (OpenCV Features)', fontsize=12, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics)
ax1.set_ylabel('Score')
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Classical ML approach
ax2 = axes[1]
for i, model_name in enumerate(['Random Forest', 'SVM', 'KNN']):
    values = [r for r in classical_results if r['Model'] == model_name][0]
    scores = [values[m] for m in metrics]
    ax2.bar(x + i * width, scores, width, label=model_name)
ax2.set_title('APPROACH 2: Classical ML (Raw Pixels)', fontsize=12, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics)
ax2.set_ylabel('Score')
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(output_dir, 'cv_vs_ml_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved chart: {chart_path}")


# ===================== VISUALIZATION 2: SIDE-BY-SIDE ACCURACY =====================
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Random Forest', 'SVM', 'KNN']
cv_acc = [r['Accuracy'] for r in cv_results]
cl_acc = [r['Accuracy'] for r in classical_results]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, cv_acc, width, label='Computer Vision (OpenCV)', color='#2ecc71')
bars2 = ax.bar(x + width/2, cl_acc, width, label='Classical ML (Raw Pixels)', color='#3498db')

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Computer Vision vs Classical ML — Accuracy Comparison',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

# Add labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
acc_chart_path = os.path.join(output_dir, 'cv_vs_ml_accuracy.png')
plt.savefig(acc_chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved accuracy chart: {acc_chart_path}")


# ===================== VISUALIZATION 3: APPROACH EXAMPLES =====================
print("\nGenerating visual examples of both approaches...")

sample_class = class_folders[0]
sample_dir = os.path.join(dataset_path, sample_class)
sample_imgs = os.listdir(sample_dir)[:1]
sample_path = os.path.join(sample_dir, sample_imgs[0])
sample_img = cv2.imread(sample_path)
sample_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Computer Vision approach
axes[0, 0].imshow(sample_rgb)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
axes[0, 1].imshow(hsv)
axes[0, 1].set_title('HSV Color Space\n(for color histograms)', fontweight='bold')
axes[0, 1].axis('off')

gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
axes[0, 2].imshow(edges, cmap='gray')
axes[0, 2].set_title('Canny Edge Detection\n(structural features)', fontweight='bold')
axes[0, 2].axis('off')

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
axes[0, 3].imshow(np.abs(laplacian), cmap='gray')
axes[0, 3].set_title('Laplacian Texture\n(disease pattern)', fontweight='bold')
axes[0, 3].axis('off')

# Row 2: Classical ML approach
axes[1, 0].imshow(sample_rgb)
axes[1, 0].set_title('Original Image', fontweight='bold')
axes[1, 0].axis('off')

resized = cv2.resize(sample_img, (32, 32))
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
axes[1, 1].imshow(resized_rgb)
axes[1, 1].set_title('Resized 32x32\n(loses detail)', fontweight='bold')
axes[1, 1].axis('off')

# Show the flattened pixel signal
flattened = resized.flatten()[:300]
axes[1, 2].plot(flattened, color='steelblue', linewidth=0.8)
axes[1, 2].set_title('Flattened Pixel Values\n(first 300 of 3072)', fontweight='bold')
axes[1, 2].set_xlabel('Pixel Index')
axes[1, 2].set_ylabel('Intensity')
axes[1, 2].grid(alpha=0.3)

# Comparison summary
axes[1, 3].axis('off')
text = ("FEATURE COUNT:\n\n"
        f"CV Approach:\n  ~86 meaningful\n  features\n\n"
        f"Classical ML:\n  3,072 raw pixel\n  values\n\n"
        f"CV uses 36x fewer\nfeatures but they're\nmore meaningful")
axes[1, 3].text(0.1, 0.5, text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Row labels
fig.text(0.02, 0.75, 'APPROACH 1:\nComputer Vision\n(OpenCV)',
         fontsize=12, fontweight='bold', color='green',
         rotation=0, va='center')
fig.text(0.02, 0.25, 'APPROACH 2:\nClassical ML\n(Raw Pixels)',
         fontsize=12, fontweight='bold', color='blue',
         rotation=0, va='center')

plt.suptitle('Computer Vision vs Classical ML — Feature Extraction',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(left=0.10)
examples_path = os.path.join(output_dir, 'approach_examples.png')
plt.savefig(examples_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved examples: {examples_path}")


# ===================== FINAL SUMMARY =====================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

best_cv = max(cv_results, key=lambda x: x['Accuracy'])
best_cl = max(classical_results, key=lambda x: x['Accuracy'])

print(f"\n🏆 Best Computer Vision Result:")
print(f"   {best_cv['Model']} — Accuracy: {best_cv['Accuracy']:.4f} ({best_cv['Accuracy']*100:.2f}%)")

print(f"\n🏆 Best Classical ML Result:")
print(f"   {best_cl['Model']} — Accuracy: {best_cl['Accuracy']:.4f} ({best_cl['Accuracy']*100:.2f}%)")

diff = (best_cv['Accuracy'] - best_cl['Accuracy']) * 100
if diff > 0:
    print(f"\n📊 Computer Vision approach is {diff:.2f}% MORE accurate")
elif diff < 0:
    print(f"\n📊 Classical ML approach is {abs(diff):.2f}% MORE accurate")
else:
    print("\n📊 Both approaches have equal accuracy")

print(f"\n📁 All results saved in: {output_dir}")
print("=" * 70)
print("\n✅ DONE! Open the output folder to see all charts and CSV results.")