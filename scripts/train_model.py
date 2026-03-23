import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset")

faces = []
labels = []

label_ids = {}
current_id = 0

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith("jpg"):

            path = os.path.join(root, file)
            label = os.path.basename(root)

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            faces.append(img)
            labels.append(id_)

labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, labels)

trainer_path = os.path.join(BASE_DIR, "trainer", "face_trainer.yml")

recognizer.save(trainer_path)

print("Training complete.")
print("Model saved at:", trainer_path)