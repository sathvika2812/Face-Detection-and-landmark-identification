import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.face_detection import detect_faces, draw_faces
from utils.landmark_detection import detect_landmarks, draw_landmarks

st.set_page_config(page_title="Face Analysis", layout="wide")
st.title("Face Detection and Landmark Identification")
st.markdown("Upload an image and choose between **Face Detection** or **Landmark Detection**")

# Sidebar
task = st.sidebar.selectbox(
    "Select Task",
    ["Face Detection", "Landmark Detection"]
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
    original_image = image.copy()

    if task == "Face Detection":
        with st.spinner("Detecting faces..."):
            faces = detect_faces(image)
            result_img = draw_faces(image.copy(), faces)
        st.success(f"✅ Detected {len(faces)} face(s)")
        
    else:  # Landmark Detection
        with st.spinner("Detecting facial landmarks..."):
            landmarks = detect_landmarks(image)
            result_img = draw_landmarks(image.copy(), landmarks)
        st.success(f"✅ Detected landmarks on {len(landmarks)} face(s)")

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(result_img, caption="Processed Image", use_column_width=True)