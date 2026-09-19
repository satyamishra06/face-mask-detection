"""
Real-time face mask detection using webcam feed.
Detects faces with OpenCV, classifies each with the trained CNN,
and draws bounding boxes + labels live.
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from face_detect import detect_faces

CLASSES = ["Mask", "No Mask", "Improper Mask"]
COLORS = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]  # green, red, orange (BGR)
CONFIDENCE_THRESHOLD = 60.0

model = load_model("models/best_model.h5")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check if another app is using it.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:
            continue

        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_rgb.astype("float32"))
        face_input = np.expand_dims(face_input, axis=0)

        pred = model.predict(face_input, verbose=0)
        idx = np.argmax(pred)
        confidence = pred[0][idx] * 100

        if confidence < CONFIDENCE_THRESHOLD:
            label = f"Uncertain ({confidence:.1f}%)"
            color = (128, 128, 128)
        else:
            label = f"{CLASSES[idx]} ({confidence:.1f}%)"
            color = COLORS[idx]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Mask Detection - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()