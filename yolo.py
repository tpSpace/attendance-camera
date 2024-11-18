import face_recognition
import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
# Initialize YOLO model for fast detection
yolo_model = YOLO("yolo-Weights/yolo11-face-custom.pt")

# Load known faces
known_faces = []
known_names = []
db_path = "db"

print("Loading face database...")
for filename in os.listdir(db_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load image and compute encoding
        image_path = os.path.join(db_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        
        # Store encoding and name
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

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
            
            # Convert YOLO box format (x1, y1, x2, y2) to face_recognition format (top, right, bottom, left)
            # This is where the main change is: the correct format is [top, right, bottom, left]
            face_locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)

    # Only process if faces are detected
    if face_locations:
        # Convert the frame to RGB format (face_recognition uses RGB)
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # Convert BGR to RGB

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through all the detected faces and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                if confidence < 0.4:
                    name = "Unknown"
        
                # Draw the rectangle and name on the frame
                (top, right, bottom, left) = face_location
                color = (0, 255, 0) if confidence > 0.4 else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (left, top-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
