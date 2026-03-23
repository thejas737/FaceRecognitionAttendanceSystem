import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cascade_path = os.path.join(BASE_DIR,"haarcascade","haarcascade_frontalface_default.xml")
trainer_path = os.path.join(BASE_DIR,"trainer","face_trainer.yml")
dataset_path = os.path.join(BASE_DIR,"dataset")

# Load face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Build label dictionary again
label_ids = {}
current_id = 0

for person in os.listdir(dataset_path):
    label_ids[current_id] = person
    current_id += 1

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

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(200,200))

        id_, confidence = recognizer.predict(face)

        if confidence < 100:
            name = label_ids.get(id_, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(frame,
                    name,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()