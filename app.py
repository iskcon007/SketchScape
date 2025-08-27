import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
MODEL_PATH = "models/best_model_mobilenet.h5"
IMG_SIZE = (224, 224)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── HELPER FUNCTIONS ──────────────────────────────────────
def load_and_prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    original_size = img.size
    if original_size != IMG_SIZE:
        img = img.resize(IMG_SIZE)
    return img, original_size

def predict_image(img, model):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array, verbose=0)[0][0]
    label = "real" if prob < 0.5 else "sketch"
    return label, prob

def get_last_conv_layer_name(model):
    """Find the last Conv2D layer in the model automatically."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("⚠️ No Conv2D layer found in model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    if not isinstance(img_array, tf.Tensor):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # If predictions is a list → unwrap it
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Auto-handle binary vs multi-class
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

def sepia_filter(img):
    sepia = np.array(img.convert("RGB"))
    sepia = cv2.transform(sepia, np.matrix([[0.393, 0.769, 0.189],
                                            [0.349, 0.686, 0.168],
                                            [0.272, 0.534, 0.131]]))
    sepia = np.clip(sepia, 0, 255)
    return Image.fromarray(sepia.astype(np.uint8))

def edge_detection(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return Image.fromarray(edges)

def invert_colors(img):
    return ImageOps.invert(img)

# ─── STREAMLIT APP ─────────────────────────────────────────
st.set_page_config(page_title="🧠 SketchScape: Real vs. Sketch Detector", layout="centered")
st.title("📸 SketchScape: Real vs. Sketch Detector")

# Load model
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("📂 Upload an image to classify:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img, original_size = load_and_prepare_image(uploaded_file)

    st.markdown(f"#### 📏 Original Size: `{original_size[0]}×{original_size[1]}` → Resized to `{IMG_SIZE[0]}×{IMG_SIZE[1]}`")
    st.image(img, caption="📸 Resized Image", use_container_width=True)

    # Prediction
    label, prob = predict_image(img, model)
    st.markdown(f"### 🔍 Prediction: **{label.upper()}**")   # only label, no probability shown

    # 🎨 IMAGE CONVERSIONS
    st.markdown("---")
    st.subheader("🎨 Image Conversion (Only for REAL images)")
    if label == "real":
        conversion = st.selectbox("Convert to:", 
                                  ["None", "Grayscale", "Color-enhance", "Sepia", "Edge Detection", "Invert Colors"],
                                  key="conversion_box")
        converted = img
        if conversion == "Grayscale":
            converted = img.convert("L")
        elif conversion == "Color-enhance":
            converted = ImageEnhance.Color(img).enhance(2.0)
        elif conversion == "Sepia":
            converted = sepia_filter(img)
        elif conversion == "Edge Detection":
            converted = edge_detection(img)
        elif conversion == "Invert Colors":
            converted = invert_colors(img)

        st.image(converted, caption=f"{conversion}", use_container_width=True)
        img_path = os.path.join(UPLOAD_DIR, f"converted_{conversion.replace(' ', '_')}.png")
        converted.save(img_path)
        with open(img_path, "rb") as f:
            st.download_button("📥 Download Converted Image", f, file_name=f"{conversion}.png")
    else:
        st.info("⚠️ Image conversion allowed only for real images.")

    # 🔥 GRAD-CAM VISUALIZATION
    st.markdown("---")
    st.subheader("🔥 Grad-CAM Visualization")
    try:
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        last_conv_layer_name = get_last_conv_layer_name(model)   # auto-detect last Conv2D
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        result = overlay_heatmap(img, heatmap)
        st.image(result, caption=f"Grad-CAM Overlay (Layer: {last_conv_layer_name})", use_container_width=True)
    except Exception as e:
        st.error(f"Grad-CAM Failed: {e}")

    # 📄 CSV Log
    st.markdown("---")
    st.subheader("📄 Save Prediction")
    log_df = pd.DataFrame([[uploaded_file.name, label]],  # only filename + label
                          columns=["filename", "label"])
    csv_path = os.path.join(UPLOAD_DIR, "prediction_log.csv")
    log_df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        st.download_button("📥 Download Prediction Log", f, file_name="prediction_log.csv")

