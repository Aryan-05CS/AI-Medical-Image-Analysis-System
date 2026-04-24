import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🧠 AI Medical Image Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pneumonia Detection using Deep Learning</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Please run training first.")
    st.stop()

model = load_model(MODEL_PATH)
classes = ['NORMAL', 'PNEUMONIA']

# ---------------- PREPROCESS ----------------
def preprocess_image(image):
    img = np.array(image)

    # Ensure 3 channels (RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    return img

# ---------------- PREDICT ----------------
def predict(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    label = classes[class_index]
    confidence = float(prediction[0][class_index])

    return label, confidence, img

# ---------------- GRAD-CAM ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + original_img

    return np.uint8(superimposed_img)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image..."):
            label, confidence, img_array = predict(image)

            # Grad-CAM
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                gradcam_image = overlay_heatmap(np.array(image), heatmap)
            except:
                gradcam_image = None

        st.markdown("---")

        # ---------------- RESULT CARD ----------------
        if label == "PNEUMONIA":
            st.markdown(
                f"""
                <div class="card">
                    <h2 style="color:#ff4b4b;">⚠️ Pneumonia Detected</h2>
                    <p>Confidence: {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="card">
                    <h2 style="color:#00c853;">✅ Normal</h2>
                    <p>Confidence: {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------- PROGRESS BAR ----------------
        st.progress(confidence)

        # ---------------- GRAD-CAM DISPLAY ----------------
        if gradcam_image is not None:
            st.markdown("### 🔥 Model Attention (Grad-CAM)")
            st.image(gradcam_image, caption="Highlighted regions used for prediction", use_column_width=True)

        # ---------------- DISCLAIMER ----------------
        st.info("⚠️ This AI tool assists diagnosis but does not replace medical professionals.")