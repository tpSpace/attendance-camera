import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import torch.nn.functional as F  # Import PyTorch functional API

# Initialize YOLO model for fast detection
yolo_model = YOLO("pretrain/yolo11-face-custom.pt")

# Load VGGFace model
vggface_model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load known faces
known_faces = []
known_names = []
db_path = "db"

print("Loading face database...")
for filename in os.listdir(db_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Load image
        image_path = os.path.join(db_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Detect faces in the image using YOLO
        results = yolo_model(image)

        # Process detections
        face_found = False
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                continue  # No detections in this result

            for box in boxes:
                if box.conf[0] < 0.5:
                    continue  # Skip low-confidence detections

                # Get coordinates from YOLO (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Crop the face from the image
                face_image = image[y1:y2, x1:x2]
                if face_image.size == 0:
                    print(f"Failed to crop face in image: {image_path}")
                    continue

                # Convert to RGB
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                # Preprocess the face image
                face_tensor = preprocess(face_image_rgb).unsqueeze(0)

                # Get the encoding
                with torch.no_grad():
                    encoding = vggface_model(face_tensor)  # Keep as tensor

                # Store encoding and name
                known_faces.append(encoding)
                known_names.append(os.path.splitext(filename)[0])

                face_found = True
                break  # Process only the first detected face

            if face_found:
                break  # Exit if face is found

        if not face_found:
            print(f"No faces found in image: {image_path}")

print(f"Loaded {len(known_names)} faces")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection for face detection
    results = yolo_model(frame)

    # Initialize the list for storing face locations
    face_locations = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.conf[0] < 0.5:  # Ignore low-confidence detections
                continue

            # Get coordinates from YOLO (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Convert YOLO box format (x1, y1, x2, y2) to (top, right, bottom, left)
            face_locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)

    # Only process if faces are detected
    if face_locations:
        # Convert the frame to RGB format (VGGFace uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face encodings
        face_encodings = []
        for (top, right, bottom, left) in face_locations:
            face_image = rgb_frame[top:bottom, left:right]
            face_tensor = preprocess(face_image).unsqueeze(0)

            with torch.no_grad():
                encoding = vggface_model(face_tensor)  # Keep as tensor

            face_encodings.append(encoding)

        # Loop through all the detected faces and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces using PyTorch cosine similarity
            similarities = []
            for known_face in known_faces:
                # Compute cosine similarity
                cosine_sim = F.cosine_similarity(face_encoding, known_face)
                similarities.append(cosine_sim.item())

            # Convert list to tensor for processing
            similarities = torch.tensor(similarities)

            # Find the index of the best match
            best_match_index = torch.argmax(similarities).item()

            # Set a threshold for cosine similarity
            threshold = 0.8  # Adjust this value based on your requirements

            # Get the best similarity score
            best_similarity = similarities[best_match_index]

            if best_similarity > threshold:
                name = known_names[best_match_index]
            else:
                name = "Unknown"

            # Draw the rectangle and name on the frame
            (top, right, bottom, left) = face_location
            color = (0, 255, 0) if best_similarity > threshold else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({best_similarity:.2f})"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()