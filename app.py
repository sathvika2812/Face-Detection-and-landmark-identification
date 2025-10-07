import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Face Detection + Landmarks", layout="wide")
st.title("Face Detection & Landmarks Identification")
st.write("Upload an image or use your webcam to detect faces and optional landmarks.")

option = st.selectbox("Select Option", ["Face Detection", "Face + Landmarks"])

# Mediapipe modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ================= Helper Function =================
def process_image(img_pil, detect_landmarks=False):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        for det in results.detections:
            mp_drawing.draw_detection(img_bgr, det)

    # Facial landmarks
    if detect_landmarks:
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                    )
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ================= Tabs =================
tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])

# --- Image Upload ---
with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        annotated_img = process_image(img, detect_landmarks=(option=="Face + Landmarks"))
        st.image(annotated_img, caption="Detected Faces / Landmarks", use_column_width=True)

# --- Webcam ---
with tab2:
    class FaceVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            annotated = process_image(img_pil, detect_landmarks=(option=="Face + Landmarks"))
            return cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

    webrtc_streamer(
        key="face-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
