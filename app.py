"""
Streamlit Web Application for Deepfake Detection - DeepShield AI v2.0
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

try:
    import cv2
except Exception:
    cv2 = None

# Import modular preprocessing and face detection
from preprocessing import preprocess_image, extract_face


# Page Configuration
st.set_page_config(
    page_title="DeepShield AI - Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism styling, gradients, and modern typography
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(0, 243, 255, 0.25);
    box-shadow: 0 8px 32px 0 rgba(0, 243, 255, 0.12);
}

.hero-banner {
    background: linear-gradient(135deg, rgba(127, 0, 255, 0.12) 0%, rgba(255, 0, 127, 0.12) 100%);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 28px;
    text-align: center;
    margin-bottom: 25px;
    backdrop-filter: blur(20px);
}

.gradient-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #FF007F 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.15rem;
    margin-top: 8px;
    margin-bottom: 16px;
}

.badge-real {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #ffffff;
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.3rem;
    display: inline-block;
    box-shadow: 0 4px 20px rgba(56, 239, 125, 0.4);
    letter-spacing: 0.5px;
}

.badge-fake {
    background: linear-gradient(135deg, #FF007F, #FF4B2B);
    color: #ffffff;
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.3rem;
    display: inline-block;
    box-shadow: 0 4px 20px rgba(255, 0, 127, 0.5);
    letter-spacing: 0.5px;
}

.progress-track {
    width: 100%;
    height: 22px;
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
    margin: 16px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.progress-fill-fake {
    height: 100%;
    background: linear-gradient(90deg, #FF007F 0%, #FF4B2B 100%);
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(255, 0, 127, 0.5);
    transition: width 1s ease;
}

.progress-fill-real {
    height: 100%;
    background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(56, 239, 125, 0.5);
    transition: width 1s ease;
}

.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #F8FAFC;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 243, 255, 0.08);
    color: #00F2FE;
    border: 1px solid rgba(0, 243, 255, 0.25);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load OpenCV Cascade Classifier for visualization safely
if cv2 is not None:
    try:
        FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    except Exception:
        face_cascade = None
else:
    face_cascade = None

@st.cache_resource
def load_detection_model(model_path):
    """Loads and caches the Keras deepfake detection model."""
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            try:
                from model import build_deepfake_detector_model
                model = build_deepfake_detector_model()
                
                if model_path.endswith('.h5') and not model_path.endswith('.weights.h5'):
                    import shutil
                    temp_weights_path = model_path + ".weights.h5"
                    shutil.copy2(model_path, temp_weights_path)
                    try:
                        model.load_weights(temp_weights_path)
                    finally:
                        if os.path.exists(temp_weights_path):
                            os.remove(temp_weights_path)
                else:
                    model.load_weights(model_path)
                return model
            except Exception as e_inner:
                st.error(f"Model Load Error: {e} (Fallback weights failed: {e_inner})")
                return None
    return None

def detect_faces_for_display(image):
    """Detects faces and overlays bounding boxes for UI feedback."""
    img_np = np.array(image.convert("RGB"))
    
    if cv2 is not None and face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_detected = len(faces) > 0
            img_display = img_np.copy()
            cropped_face = None
            
            for (x, y, w, h) in faces:
                # Draw glowing turquoise rectangle
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 243, 255), 3)
                cv2.putText(img_display, "FACIAL REGION DETECTED", (x, max(y-10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 243, 255), 2)
                if cropped_face is None:
                    # Crop face for zoom box
                    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                    x2, y2 = min(img_np.shape[1], x + w + pad_w), min(img_np.shape[0], y + h + pad_h)
                    cropped_face = img_np[y1:y2, x1:x2]
                    
            if cropped_face is None:
                cropped_face = cv2.resize(img_np, (300, 300))
            else:
                cropped_face = cv2.resize(cropped_face, (300, 300))
                
            return Image.fromarray(img_display), Image.fromarray(cropped_face), face_detected
        except Exception:
            pass
            
    # PIL Fallback if cv2 is not available or encounters an error
    img_pil = image.convert("RGB")
    cropped_pil = img_pil.resize((300, 300))
    return img_pil, cropped_pil, False


# Header Section
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: center; margin-bottom: 10px;">
        <span class="status-pill">⚡ Neural Network Active</span>
    </div>
    <h1 class="gradient-title">🛡️ DEEPSHIELD AI v2.0</h1>
    <p class="hero-subtitle">Next-Gen Facial Artifact & Deepfake Manipulation Analyzer</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.markdown('### ⚙️ Engine Configuration')

model_selection_options = {
    "EfficientNetB3 Best (.keras)": os.path.join("saved_model", "deepfake_detector_best.keras"),
    "EfficientNetB3 Legacy (.h5)": os.path.join("saved_model", "deepfake_detector_best.h5")
}
selected_model_name = st.sidebar.selectbox("Select Model Format", list(model_selection_options.keys()))
model_path = model_selection_options[selected_model_name]

# System Health & Hardware Info
gpus = tf.config.list_physical_devices('GPU')
gpu_status = f"🟢 GPU Accelerated ({len(gpus)} device)" if gpus else "⚪ CPU Multi-threaded"

st.sidebar.markdown(f"""
---
### 📊 System Telemetry
- **Hardware:** {gpu_status}
- **Input Tensor:** 300x300 RGB
- **Backbone:** EfficientNetB3
- **Detector:** MediaPipe / OpenCV Cascade
""")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Pro Tip:** Upload portrait images with clear lighting for maximum face detection accuracy.
""")

# Load Model
model = load_detection_model(model_path)

# Main Multi-Tab Navigation Layout
tab1, tab2, tab3 = st.tabs(["🔍 Image Scanner", "📈 Analytics & Performance", "ℹ️ System Architecture"])

with tab1:
    col_input, col_preset = st.columns([2, 1])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("1. Select or Upload Image")
        uploaded_file = st.file_uploader(
            "Drop your image file below (JPEG, PNG, WEBP)", 
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_preset:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Quick Test Samples")
        st.write("Or pick a sample dataset image:")
        
        # Scan for existing sample files in dataset/
        real_samples = [os.path.join("dataset", "real", f) for f in os.listdir(os.path.join("dataset", "real")) if f.endswith(('.jpg', '.png'))] if os.path.exists(os.path.join("dataset", "real")) else []
        fake_samples = [os.path.join("dataset", "fake", f) for f in os.listdir(os.path.join("dataset", "fake")) if f.endswith(('.jpg', '.png'))] if os.path.exists(os.path.join("dataset", "fake")) else []
        
        sample_choice = None
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if real_samples and st.button("Sample REAL", use_container_width=True):
                sample_choice = real_samples[0]
        with col_s2:
            if fake_samples and st.button("Sample FAKE", use_container_width=True):
                sample_choice = fake_samples[0]
        st.markdown('</div>', unsafe_allow_html=True)

    # Determine image source
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
    elif sample_choice is not None and os.path.exists(sample_choice):
        active_image = Image.open(sample_choice)
        
    if active_image is not None:
        annotated_img, cropped_face_img, face_detected = detect_faces_for_display(active_image)
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Image Analysis & Face Crop")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.write("**Full Image + Bounding Box**")
                st.image(annotated_img, use_container_width=True)
            with subcol2:
                st.write("**Cropped Facial Tensor**")
                st.image(cropped_face_img, use_container_width=True)
                
            if face_detected:
                st.success("🎯 Facial ROI extracted successfully.")
            else:
                st.info("ℹ️ No face detected. Evaluated full image layout.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Evaluation Dashboard")
            
            if model is None:
                st.warning("⚠️ Trained model weights not found. Please train model using train.py first.")
            else:
                if st.button("🚀 EXECUTE NEURAL INFERENCE", type="primary", use_container_width=True):
                    # Save temp image for processing
                    temp_img_path = "temp_scan.jpg"
                    active_image.convert("RGB").save(temp_img_path)
                    
                    with st.spinner("Analyzing high-frequency artifacts & feature maps..."):
                        try:
                            # Preprocess and inference
                            processed_tensor = preprocess_image(temp_img_path, target_size=(300, 300), is_training=False)
                            tensor_input = np.expand_dims(processed_tensor, axis=0)
                            
                            prediction_prob = float(model.predict(tensor_input, verbose=0)[0][0])
                            
                            # Clean temp file
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                                
                            st.markdown("---")
                            
                            if prediction_prob >= 0.5:
                                confidence = prediction_prob * 100.0
                                st.markdown(f'<div style="text-align: center; margin-bottom: 15px;"><span class="badge-fake">⚠️ FAKE DETECTED</span></div>', unsafe_allow_html=True)
                                st.markdown(f"**Deepfake Probability:** `{confidence:.2f}%`")
                                st.markdown(f"""
                                <div class="progress-track">
                                    <div class="progress-fill-fake" style="width: {confidence}%;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.warning("🚨 High likelihood of digital face manipulation or synthetic generation.")
                            else:
                                confidence = (1.0 - prediction_prob) * 100.0
                                st.markdown(f'<div style="text-align: center; margin-bottom: 15px;"><span class="badge-real">✅ REAL IMAGE</span></div>', unsafe_allow_html=True)
                                st.markdown(f"**Authenticity Confidence:** `{confidence:.2f}%`")
                                st.markdown(f"""
                                <div class="progress-track">
                                    <div class="progress-fill-real" style="width: {confidence}%;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.success("✅ Image features match authentic camera sensor characteristics.")
                                
                            # Diagnostic Metrics Cards
                            st.markdown("---")
                            st.write("##### Diagnostic Telemetry:")
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Raw Output</div>
                                    <div class="metric-value">{prediction_prob:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with m2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">ROI Status</div>
                                    <div class="metric-value">{'Face' if face_detected else 'Full'}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with m3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Confidence</div>
                                    <div class="metric-value">{confidence:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Inference Failure: {e}")
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                else:
                    st.info("Click the button above to execute CNN feature map extraction.")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Model Training & Evaluation Metrics")
    
    saved_model_dir = "saved_model"
    metrics_json_path = os.path.join(saved_model_dir, "evaluation_results.json")
    
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            metrics_data = json.load(f)
            
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Test Loss", f"{metrics_data.get('test_loss', 0):.4f}")
        with col_m2:
            st.metric("Accuracy", f"{metrics_data.get('test_accuracy', 0)*100:.2f}%")
        with col_m3:
            st.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
        with col_m4:
            st.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
        with col_m5:
            st.metric("F1-Score", f"{metrics_data.get('f1_score', 0):.4f}")
            
    st.markdown("---")
    
    col_img1, col_img2, col_img3 = st.columns(3)
    
    history_img = os.path.join(saved_model_dir, "training_history_curves.png")
    cm_img = os.path.join(saved_model_dir, "confusion_matrix.png")
    roc_img = os.path.join(saved_model_dir, "roc_curve.png")
    
    with col_img1:
        st.write("**Training History**")
        if os.path.exists(history_img):
            st.image(history_img, use_container_width=True)
        else:
            st.caption("Training history curves plot unavailable.")
            
    with col_img2:
        st.write("**Confusion Matrix**")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)
        else:
            st.caption("Confusion matrix plot unavailable.")
            
    with col_img3:
        st.write("**ROC Curve**")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)
        else:
            st.caption("ROC curve plot unavailable.")
            
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ DeepShield AI Pipeline Architecture")
    
    st.markdown("""
    ### Pipeline Stages
    
    1. **Facial Region of Interest (ROI) Detection**:
       - Primary: MediaPipe Face Mesh / Detection model.
       - Fallback: OpenCV Haar Cascade frontal face classifier.
       - Extracts facial bounding box with 15% padding to capture hairline & jawline boundary artifacts.
       
    2. **Normalization & Rescaling**:
       - Resized to 300x300 RGB input tensor.
       - Pixel values normalized to float32 [0.0, 1.0] range.
       
    3. **EfficientNetB3 Transfer Learning Backbone**:
       - Pre-trained on ImageNet.
       - Global Average Pooling collapses 10x10x1536 feature maps.
       - Dense classification head (256 units + Batch Normalization + ReLU + Dropout 0.4).
       - Sigmoid scalar output representing probability of synthetic manipulation.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
