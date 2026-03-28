import cv2
import numpy as np
import os

dataset_dir = "faces"
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels = []
faces_data = []

label_id = 0
label_dict = {}  # Map label id -> user name

# Make sure the dataset folder exists
os.makedirs(dataset_dir, exist_ok=True)

for user_name in os.listdir(dataset_dir):
    user_dir = os.path.join(dataset_dir, user_name)
    label_dict[label_id] = user_name

    for file in os.listdir(user_dir):
        img = cv2.imread(os.path.join(user_dir, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces_data.append(img)
            labels.append(label_id)
    label_id += 1

if faces_data:  # Check if we have any images
    recognizer.train(faces_data, np.array(labels))
    # Save the trained model
    # Change this line in train_model.py
    recognizer.save(r"C:\Users\KIRAN S\vs\face_model.yml")
    
    np.save(r"C:\Users\KIRAN S\vs\labels.npy", label_dict)
    print("Training complete ✅")
else:
    print("No face images found! Please register users first.")