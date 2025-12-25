import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(
    page_title="Multi-Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Multi-Digit Handwritten Recognition")

# Load model once
@st.cache_resource
def load_model():
    # Note: Ensure "digit_model.h5" is in the same directory
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

# Sidebar instructions
st.sidebar.title("ℹ️ Instructions")
st.sidebar.markdown("""
- Upload an image of handwritten digits.
- The app uses **Otsu's Binarization** to handle lighting.
- Results are sorted from **left to right**.
""")

show_debug = st.sidebar.checkbox("Show Debug Processing", value=False)

# Upload image
uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)
    st.image(img_np, caption="Uploaded Image", width=400)

    if st.button("Predict"):
        # 1. PREPROCESSING: Reduce grain and invert
        img_inv = 255 - img_np
        img_blur = cv2.GaussianBlur(img_inv, (5, 5), 0)

        # 2. AUTO-THRESHOLD: Otsu's method automatically finds the best cut-off
        _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. CLEANING: Dilation closes small gaps in pen strokes
        kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.dilate(img_bin, kernel, iterations=1)

        if show_debug:
            st.image(img_bin, caption="Processed Binary Image (What the AI sees)", width=400)

        # 4. CONTOUR DETECTION
        contours, _ = cv2.findContours(
            img_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. FILTERING: Ignore tiny noise
        digit_boxes = []
        # Dynamic threshold: ignores objects smaller than 0.1% of the image
        min_area_threshold = (img_np.shape[0] * img_np.shape[1]) * 0.001 
        
        for c in contours:
            if cv2.contourArea(c) > min_area_threshold:
                x, y, w, h = cv2.boundingRect(c)
                digit_boxes.append((x, y, w, h))

        if not digit_boxes:
            st.warning("⚠️ No digits detected. Ensure high contrast.")
            st.stop()

        # Sort digits left-to-right
        digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

        predicted_digits = []
        cols = st.columns(len(digit_boxes)) # Create columns for each digit display

        for i, (x, y, w, h) in enumerate(digit_boxes):
            # Extract and Pad
            digit_img = img_bin[y:y+h, x:x+w]
            digit_img = cv2.copyMakeBorder(
                digit_img, 15, 15, 15, 15,
                cv2.BORDER_CONSTANT, value=0
            )

            # Centering logic
            h_pad, w_pad = digit_img.shape
            y_coords, x_coords = np.where(digit_img > 0)
            if len(y_coords) > 0:
                cy, cx = np.mean(y_coords), np.mean(x_coords)
                shift_x = w_pad//2 - int(cx)
                shift_y = h_pad//2 - int(cy)
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                digit_img = cv2.warpAffine(digit_img, M, (w_pad, h_pad))

            # Resize & Predict
            digit_resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            img_norm = digit_resized / 255.0
            img_norm = img_norm.reshape(1, 28, 28, 1)

            prediction = model.predict(img_norm)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_digits.append(str(digit))

            with cols[i]:
                st.image(digit_resized, caption=f"ID:{i+1}", width=60)
                st.caption(f"**{digit}** ({confidence:.0f}%)")

        # Result display
        final_number = ''.join(predicted_digits)
        st.success(f"### Predicted Number: {final_number}")