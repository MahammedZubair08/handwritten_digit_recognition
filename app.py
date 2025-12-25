import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

# ---------------- CUSTOM RESPONSIVE CSS ----------------
st.markdown("""
<style>

/* Main container */
.main {
    padding: 1rem;
}

/* Title */
h1 {
    text-align: center;
    font-size: 2.2rem;
}

/* Uploaded & processed images */
.responsive-img img {
    width: 100%;
    max-width: 420px;
    height: auto;
    display: block;
    margin: auto;
}

/* Predict button */
.stButton>button {
    width: 100%;
    font-size: 1.1rem;
    padding: 0.6rem;
}

/* Digit card */
.digit-card {
    text-align: center;
    padding: 0.5rem;
    border-radius: 10px;
    background-color: #f7f7f7;
}

/* Mobile optimizations */
@media (max-width: 768px) {
    h1 {
        font-size: 1.7rem;
    }
    .digit-card {
        margin-bottom: 0.7rem;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("✍️ Multi-Digit Handwritten Recognition")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "DigitClassifier.keras",
        compile=False
    )

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ Instructions")
st.sidebar.markdown("""
- Upload an image of handwritten digits  
- Uses **Otsu’s Thresholding**  
- Digits are read **left → right**
""")

show_debug = st.sidebar.checkbox("Show Debug Processing", value=False)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    st.markdown('<div class="responsive-img">', unsafe_allow_html=True)
    st.image(img_np, caption="Uploaded Image")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict Digits"):
     

        # -------- PREPROCESSING --------
        img_inv = 255 - img_np   
        img_blur = cv2.GaussianBlur(img_inv, (5, 5), 0)

        _, img_bin = cv2.threshold(
            img_blur, 0, 255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.dilate(img_bin, kernel, iterations=1)

        if show_debug:
            st.markdown('<div class="responsive-img">', unsafe_allow_html=True)
            st.image(img_bin, caption="Processed Image")
            st.markdown('</div>', unsafe_allow_html=True)

        # -------- CONTOURS --------
        contours, _ = cv2.findContours(
            img_bin.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = img_np.shape[0] * img_np.shape[1] * 0.001
        boxes = []

        for c in contours:
            if cv2.contourArea(c) > min_area:
                boxes.append(cv2.boundingRect(c))

        if not boxes:
            st.warning("No digits detected. Use high contrast.")
            st.stop()

        boxes = sorted(boxes, key=lambda b: b[0])

        predicted_digits = []

        # -------- RESPONSIVE DIGIT GRID --------
        is_mobile = len(boxes) <= 2
        cols = st.columns(1 if is_mobile else min(len(boxes), 6))

        for i, (x, y, w, h) in enumerate(boxes):

            digit_img = img_bin[y:y+h, x:x+w]
            digit_img = cv2.copyMakeBorder(
                digit_img, 15, 15, 15, 15,
                cv2.BORDER_CONSTANT, value=0
            )

            ys, xs = np.where(digit_img > 0)
            if len(xs) > 0:
                cx, cy = xs.mean(), ys.mean()
                M = np.float32([
                    [1, 0, digit_img.shape[1]//2 - cx],
                    [0, 1, digit_img.shape[0]//2 - cy]
                ])
                digit_img = cv2.warpAffine(
                    digit_img, M,
                    (digit_img.shape[1], digit_img.shape[0])
                )

            digit_resized = cv2.resize(digit_img, (28, 28))
            img_norm = digit_resized / 255.0
            img_norm = img_norm.reshape(1, 28, 28, 1)

            pred = model.predict(img_norm, verbose=0)
            digit = np.argmax(pred)
            conf = np.max(pred) * 100
            predicted_digits.append(str(digit))

            with cols[i % len(cols)]:
                st.markdown('<div class="digit-card">', unsafe_allow_html=True)
                st.image(digit_resized, width=80)
                st.markdown(f"**{digit}** ({conf:.0f}%)")
                st.markdown('</div>', unsafe_allow_html=True)

        # -------- FINAL RESULT --------
        final_number = "".join(predicted_digits)
        st.success(f"### Predicted Number: {final_number}")
