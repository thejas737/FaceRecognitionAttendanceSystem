import cv2
import os
import numpy as np
import pickle

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset")
trainer_dir = os.path.join(BASE_DIR, "trainer")

faces = []
labels = []

label_ids = {}
current_id = 0

# Step 1: Traverse dataset and build labels
for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".jpg"):

            path = os.path.join(root, file)
            label = os.path.basename(root)

            # Assign ID to each person
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # Read image in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue  # skip corrupted images

            faces.append(img)
            labels.append(id_)

# Convert labels to numpy array
labels = np.array(labels)

# Step 2: Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Step 3: Save trained model
trainer_path = os.path.join(trainer_dir, "face_trainer.yml")
recognizer.save(trainer_path)

# Step 4: Save label mapping (THIS WAS YOUR BUG)
labels_path = os.path.join(trainer_dir, "labels.pickle")

with open(labels_path, 'wb') as f:
    pickle.dump(label_ids, f)

print("Training complete.")
print("Model saved at:", trainer_path)
print("Labels saved at:", labels_path)
print("Label mapping:", label_ids)