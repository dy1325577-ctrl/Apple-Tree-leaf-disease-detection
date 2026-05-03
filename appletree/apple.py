from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# ============================================================
# --- 1. SETTINGS ---
# ============================================================
# Original (Classical) approach uses 32x32 + RGB histograms
image_size = (32, 32)
# CV approach uses 64x64 + HSV histograms + edges + texture
CV_IMG_SIZE = (64, 64)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# --- 2. LOAD ORIGINAL (CLASSICAL) MODELS ---
# ============================================================
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'apple_disease_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    anomaly_detector = joblib.load(os.path.join(MODEL_DIR, 'anomaly_detector.pkl'))
    print("✅ Original (Classical) models loaded successfully.")
    classical_loaded = True
except Exception as e:
    print(f"❌ Error loading Classical models: {e}")
    model = scaler = pca = label_encoder = anomaly_detector = None
    classical_loaded = False


# ============================================================
# --- 2b. LOAD NEW CV MODELS ---
# ============================================================
try:
    cv_model = joblib.load(os.path.join(MODEL_DIR, 'cv_apple_disease_model.pkl'))
    cv_scaler = joblib.load(os.path.join(MODEL_DIR, 'cv_scaler.pkl'))
    cv_pca = joblib.load(os.path.join(MODEL_DIR, 'cv_pca.pkl'))
    cv_label_encoder = joblib.load(os.path.join(MODEL_DIR, 'cv_label_encoder.pkl'))
    cv_anomaly_detector = joblib.load(os.path.join(MODEL_DIR, 'cv_anomaly_detector.pkl'))
    print("✅ New CV (Computer Vision) models loaded successfully.")
    cv_loaded = True
except Exception as e:
    print(f"⚠️  CV models not found: {e}")
    print("   Run train_cv_models.py first to enable CV mode.")
    cv_model = cv_scaler = cv_pca = cv_label_encoder = cv_anomaly_detector = None
    cv_loaded = False


# ============================================================
# --- 3. FEATURE EXTRACTION ---
# ============================================================
def extract_features_classical(image_path):
    """
    Original/Classical approach: RGB Histograms (96 features).
    This MUST match the original training script.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, image_size)
    
    hist_red   = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_green = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_blue  = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    
    return np.concatenate([hist_red, hist_green, hist_blue])


def extract_features_cv(image_path):
    """
    Advanced CV approach: HSV Histograms + Canny Edges + Laplacian + Stats (86 features).
    This MUST match train_cv_models.py exactly.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    features = []
    image = cv2.resize(image, CV_IMG_SIZE)
    
    # 1. HSV Color Histogram (72 features)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [24], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # 2. Canny Edge Density (1 feature)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # 3. Channel Statistics (12 features)
    for channel in range(3):
        features.append(np.mean(image[:, :, channel]))
        features.append(np.std(image[:, :, channel]))
    for channel in range(3):
        features.append(np.mean(hsv[:, :, channel]))
        features.append(np.std(hsv[:, :, channel]))
    
    # 4. Texture - Laplacian Variance (1 feature)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.var())
    
    return np.array(features)


# ============================================================
# --- 4. TREATMENT RECOMMENDATIONS ---
# ============================================================
treatments = {
    'Apple___Apple_scab': {
        'description': "Fungal disease causing olive-green to black spots on leaves and fruit",
        'treatment': [
            "Chemical: Apply sulfur, myclobutanil (Rally), or fenarimol (Rubigan) fungicides",
            "Organic: Remove fallen leaves, prune for air circulation, use sulfur sprays",
            "Timing: Treat from green tip through first cover spray period",
            "Resistant varieties: 'Liberty', 'Freedom'"
        ],
        'prevention': [
            "Rake and destroy fallen leaves in autumn",
            "Maintain proper tree spacing",
            "Avoid overhead irrigation",
            "Apply dormant sprays in late winter"
        ]
    },
    'Apple___Black_rot': {
        'description': "Fungal disease causing fruit rot and leaf spots",
        'treatment': [
            "Chemical: Use captan, thiophanate-methyl, or mancozeb fungicides",
            "Organic: Remove mummified fruit, prune cankers, use copper sprays",
            "Timing: Begin at petal fall, continue every 10-14 days in wet weather",
            "Critical: Remove infected material immediately"
        ],
        'prevention': [
            "Prune out dead wood 12 inches below cankers",
            "Disinfect pruning tools between cuts",
            "Avoid fruit wounding during handling",
            "Store fruit at 32°F"
        ]
    },
    'Apple___Cedar_apple_rust': {
        'description': "Fungal disease requiring both apple and cedar hosts",
        'treatment': [
            "Chemical: Apply myclobutanil, fenarimol, or trifloxystrobin fungicides",
            "Organic: Remove nearby junipers, use sulfur or copper sprays",
            "Timing: Start at pink bud stage, continue every 7-10 days until petal fall",
            "Resistant varieties: 'Redfree', 'Liberty'"
        ],
        'prevention': [
            "Eliminate junipers within 2 miles if possible",
            "Rake and destroy fallen leaves",
            "Apply dormant oil sprays",
            "Select resistant varieties"
        ]
    },
    'Healthy': {
        'description': "No signs of disease detected",
        'treatment': [
            "Continue regular monitoring",
            "Maintain proper fertilization and irrigation",
            "Prune annually for air circulation",
            "Consider preventive fungicides if disease pressure is high"
        ],
        'prevention': []
    }
}


# ============================================================
# --- 5. SHARED PREDICTION HELPER ---
# ============================================================
def make_prediction(features, model_obj, scaler_obj, pca_obj,
                    label_encoder_obj, anomaly_detector_obj):
    """Common prediction pipeline used by both endpoints."""
    features_scaled = scaler_obj.transform([features])
    features_pca = pca_obj.transform(features_scaled)
    
    is_anomaly = anomaly_detector_obj.predict(features_pca)[0] == -1
    
    if is_anomaly:
        return {
            'is_anomaly': True,
            'message': "Unknown Image - This doesn't appear to be an apple leaf or shows symptoms not recognized by our system",
            'recommendations': [
                "Verify the image is of an apple leaf",
                "Check for image quality issues",
                "Consult with a plant pathologist",
                "Compare with known apple disease references"
            ]
        }
    
    pred = model_obj.predict(features_pca)[0]
    disease = label_encoder_obj.inverse_transform([pred])[0]
    
    disease_info = treatments.get(disease, {
        'description': f"Unknown disease category: {disease}",
        'treatment': ["Consult local agricultural extension"],
        'prevention': ["Maintain good plant health"]
    })
    
    return {
        'is_anomaly': False,
        'disease': disease,
        'description': disease_info['description'],
        'treatment': disease_info['treatment'],
        'prevention': disease_info['prevention']
    }


# ============================================================
# --- 6. FLASK ROUTES ---
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/status')
def status():
    """Returns which approaches are available."""
    return jsonify({
        'classical_available': classical_loaded,
        'cv_available': cv_loaded,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts an optional 'mode' parameter:
      - 'classical' (default) -> Uses original RGB histogram approach
      - 'cv'                  -> Uses new advanced CV approach
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Determine which mode to use
    mode = request.form.get('mode', 'classical').lower()
    
    if mode == 'cv':
        if not cv_loaded:
            return jsonify({'error': 'CV models not loaded. Run train_cv_models.py first.'})
    else:
        if not classical_loaded:
            return jsonify({'error': 'Classical models not loaded.'})
    
    filename = secure_filename(file.filename)
    temp_dir = tempfile.gettempdir()
    temp_filepath = os.path.join(temp_dir, filename)
    
    try:
        file.save(temp_filepath)
        
        if mode == 'cv':
            features = extract_features_cv(temp_filepath)
            if features is None:
                return jsonify({'error': 'Could not process image'})
            
            result = make_prediction(
                features, cv_model, cv_scaler, cv_pca,
                cv_label_encoder, cv_anomaly_detector
            )
            result['mode'] = 'Computer Vision (HSV + Canny + Laplacian + Stats)'
            result['features_used'] = 86
        else:
            features = extract_features_classical(temp_filepath)
            if features is None:
                return jsonify({'error': 'Could not process image'})
            
            result = make_prediction(
                features, model, scaler, pca,
                label_encoder, anomaly_detector
            )
            result['mode'] = 'Classical (RGB Histograms)'
            result['features_used'] = 96
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})
    
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


@app.route('/compare', methods=['POST'])
def compare():
    """
    Runs BOTH approaches on the same image and returns both results
    side-by-side - perfect for your report comparison!
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not (classical_loaded and cv_loaded):
        return jsonify({
            'error': 'Both classical and CV models must be loaded for comparison. '
                     'Run train_cv_models.py first.'
        })
    
    filename = secure_filename(file.filename)
    temp_dir = tempfile.gettempdir()
    temp_filepath = os.path.join(temp_dir, filename)
    
    try:
        file.save(temp_filepath)
        
        # Classical prediction
        classical_features = extract_features_classical(temp_filepath)
        classical_result = make_prediction(
            classical_features, model, scaler, pca,
            label_encoder, anomaly_detector
        )
        
        # CV prediction
        cv_features = extract_features_cv(temp_filepath)
        cv_result = make_prediction(
            cv_features, cv_model, cv_scaler, cv_pca,
            cv_label_encoder, cv_anomaly_detector
        )
        
        return jsonify({
            'classical': {
                'mode': 'Classical (RGB Histograms)',
                'features_used': 96,
                **classical_result
            },
            'cv': {
                'mode': 'Computer Vision (HSV + Canny + Laplacian + Stats)',
                'features_used': 86,
                **cv_result
            },
            'agreement': (
                classical_result.get('disease') == cv_result.get('disease')
                if not (classical_result.get('is_anomaly') or cv_result.get('is_anomaly'))
                else None
            )
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})
    
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


# ============================================================
# --- 7. RUN THE APP ---
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5007))
    print("\n" + "=" * 60)
    print("🍎 Apple Leaf Disease Detection - Web App")
    print("=" * 60)
    print(f"  Classical Mode (RGB Histograms): "
          f"{'✅ Ready' if classical_loaded else '❌ Not loaded'}")
    print(f"  CV Mode (HSV+Edges+Texture):     "
          f"{'✅ Ready' if cv_loaded else '❌ Not loaded'}")
    print("=" * 60)
    print(f"  Web URL: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)