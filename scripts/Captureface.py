import cv2
import os

# Locate project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Haar cascade path
cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Ask user name
person_name = input("Enter person's name: ").title()

# Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset", person_name)

# Create folder if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 80

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

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize for consistency
        face = cv2.resize(face, (200, 200))

        count += 1

        # Save image
        file_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_path, face)

        # Draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.putText(frame,
                    f"Images Captured: {count}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.imshow("Face Capture", frame)

    # Stop when enough images captured
    if count >= max_images:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Face capture complete.")