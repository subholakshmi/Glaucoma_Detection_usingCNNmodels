import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model
model = load_model("best_glaucoma_hybrid_model.h5", compile=False)

# ⚠️ Adjust based on your training
class_names = ["Glaucoma", "Normal"]   # ✅ corrected order

st.title("Glaucoma Detection App with Grad-CAM")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

# ✅ Preprocess (FIXED SIZE)
def preprocess(img):
    img = img.resize((256, 256))   # ✅ MUST match training
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 🔥 Grad-CAM
def get_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap  # ✅ FIXED

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    return np.uint8(superimposed_img)

# 🚀 Main Logic
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Fundus Image", use_container_width=True)

    img_array = preprocess(img)

    prediction = model.predict(img_array)

    # ✅ Handle both sigmoid and softmax
    if prediction.shape[1] == 1:
        # Binary sigmoid
        prob = prediction[0][0]
        if prob > 0.5:
            pred_class = "Glaucoma"
            confidence = prob
        else:
            pred_class = "Normal"
            confidence = 1 - prob
    else:
        # Softmax
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.4f}")

    # 🔥 Grad-CAM
    try:
        heatmap = get_gradcam(model, img_array, "conv5_block16_2_conv")

        original = np.array(img.resize((256, 256)))
        overlay = overlay_heatmap(original, heatmap)

        st.subheader("Grad-CAM Visualization")
        st.image(overlay, caption="Model Attention Heatmap", use_container_width=True)

    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")