import cv2
import numpy as np
import webbrowser
import os
web = input("enter your url:")
# Paths to trained model and labels
model_path = r"C:\Users\KIRAN S\vs\face_model.yml"
labels_path = r"C:\Users\KIRAN S\vs\labels.npy"

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("Model or labels file not found! Run training script first.")
    exit()

# Load face recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_dict = np.load(labels_path, allow_pickle=True).item()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

print("Looking for authorized users...")

fail_count = 0  # Initialize failure counter

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)

        if confidence < 70:  # Recognized
            print(f"Access Granted: {label_dict[label]}")
            webbrowser.open(f"https://www.{web}.com")  # Replace with your URL
            cap.release()
            cv2.destroyAllWindows()
            exit()
        else:  # Not recognized
            fail_count += 1
            cv2.putText(frame, "Unauthorized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            print(f"Unauthorized face detected ❌ ({fail_count}/4)")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # If fails more than 4 times, stop
    if fail_count > 4:
        print("Too many failed attempts. Exiting...")
        break

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()