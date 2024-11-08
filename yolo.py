from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO("yolo-Weights/yolo11-face-custom.pt")  # Load your custom face detection model

# Object classes
classNames = ["face"]

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class index
            cls = int(box.cls[0])
            # Check if detected class is "face"
            if classNames[cls] == "face":
                if box.conf[0] < 0.5:
                    continue
                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw rectangle around face
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # Display class name and confidence
                label = f"{classNames[cls]} {confidence}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()