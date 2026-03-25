import cv2
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cascade_path = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")
trainer_path = os.path.join(BASE_DIR, "trainer", "face_trainer.yml")
labels_path = os.path.join(BASE_DIR, "trainer", "labels.pickle")

# Load face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Load label mapping
if not os.path.exists(labels_path) or os.path.getsize(labels_path) == 0:
    print("labels.pickle missing or empty. Please retrain the model.")
    exit()

with open(labels_path, 'rb') as f:
    original_labels = pickle.load(f)

# Reverse mapping (ID → Name)
label_ids = {v: k for k, v in original_labels.items()}

# Start camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        id_, confidence = recognizer.predict(face)

        # Lower confidence = better match
        if confidence < 100:
            name = label_ids.get(id_, "Unknown")
        else:
            name = "Unknown"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display name
        cv2.putText(
            frame,
            f"{name} ({round(confidence, 2)})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()