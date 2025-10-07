import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks(image):
    """Detect facial landmarks using MediaPipe Face Mesh"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                coords = []
                for landmark in face_landmarks.landmark:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    coords.append((px, py))
                landmarks_list.append(np.array(coords))
        return landmarks_list

def draw_landmarks(image, landmarks_list):
    """Draw landmarks on image"""
    for landmarks in landmarks_list:
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
    return image