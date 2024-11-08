import os
import time
from typing import List, Optional, Tuple
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from ultralytics import YOLO
from utils import Ok, Result, Err

class Person:
    def __init__(self, name: str, embedding: np.ndarray):
        self.name = name
        self.embedding = embedding

    def __iter__(self):
        return iter((self.name, self.embedding))

class FaceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initializes the YOLO face detector.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.classNames = ["face"] 

    def detect_faces(self, img: np.ndarray):
        """
        Detect faces in an image using YOLO.
        Returns a list of bounding boxes for detected faces.
        """
        results = self.model(img)
        detected_faces = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.classNames[cls] == "face" and box.conf[0] >= self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    detected_faces.append((x1, y1, x2, y2, box.conf[0].item()))
        
        return detected_faces
    
    def get_bounding_box(self, img: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get bounding box for detected faces in an image.
        """
        results = self.model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.classNames[cls] == "face" and box.conf[0] >= self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    return ((x1, y1), (x2, y2))
        

class FaceRecognizer:
    def __init__(self, known_faces_dir: str, model=None):
        """
        Initializes the VGG model for face recognition if not provided.
        Parameters:
        - known_faces_dir: Directory containing known faces.
        - model: Pretrained VGG model for face recognition
        """
        if model is None:
            self.model = models.vgg16(pretrained=True)
            self.model.eval()
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) 
        else:
            self.model = model 

        self.known_face: List[Person] = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load known faces
        self.load_known_faces(known_faces_dir)

    def extract_face_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extracts the face embedding from the VGG model.
        """
        pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.squeeze().numpy() 

    def load_known_faces(self, known_faces_dir: str) -> Result[None]:
        """
        Load known face images from the directory and extract embeddings.
        """
        for person_name in os.listdir(known_faces_dir):
            person_folder = os.path.join(known_faces_dir, person_name)
            
            if os.path.isdir(person_folder):
                for image_file in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, image_file)
                    image = cv2.imread(image_path)

                    if image is not None:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        embedding = self.extract_face_embedding(rgb_image)
                        self.known_face.append(Person(person_name, embedding))

                    else:
                        return Err(f"Could not read image: {image_path}")


    def recognize_face(self, face_embedding: np.ndarray, threshold: int = 0.8) -> str:
        """
        Compares the extracted face embedding with known faces.
        Returns the name of the recognized person.
        """
        matches = []
        for _, known_embedding in self.known_face:
            similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
            matches.append(similarity)
        if max(matches) > threshold:
            match_idx = matches.index(max(matches))
            return self.known_face[match_idx].name
        else:
            return "Unknown"


class FaceRecognitionApp:
    def __init__(self, face_detector: FaceDetector, face_recognizer: FaceRecognizer):
        """
        Initializes the Face Recognition application with face detector and recognizer.
        """
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

    def run(self) -> Result[None]:
        """
        Main loop for detecting and recognizing faces.
        """
        while True:
            success, img = self.cap.read()
            if not success:
                return Err("Could not read frame")
            _, err = self._process_frame(img)
            if err:
                return Err(err)
            if cv2.waitKey(1) == ord('q'):
                print("Quitting...")
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return Ok(None)

    def _process_frame(self, img: np.ndarray) -> Result[None]:
        """
        Process the frame to detect and recognize faces
        """
        detected_faces: List[Tuple[int, int, int, int, float]] = self.face_detector.detect_faces(img)
        for x1, y1, x2, y2, confidence in detected_faces:
            face_img: np.ndarray = img[y1:y2, x1:x2]
            face_embedding: np.ndarray = self.face_recognizer.extract_face_embedding(face_img)
            name: str = self.face_recognizer.recognize_face(face_embedding=face_embedding, threshold=0.6)
            self._draw_label(img, x1, y1, x2, y2, name, confidence)
        cv2.imshow('Face Recognition', img)
        return Ok(None)

    def _draw_label(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, name: str, confidence: float) -> None:
        """
        Draw bounding box and label the name on the image.
        """
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label: str = f"{name} ({confidence:.2f})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    def _cleanup(self):
        """
        Cleanup the resources.
        """
        self.cap.release()
        cv2.destroyAllWindows()

class FaceRegister:
    def __init__(self, face_detector: 'FaceDetector', save_dir: str, n: int = 5):
        """
        Initializes the face registration system.

        :param face_detector: The face detector object (expected to have a method get_bounding_box)
        :param save_dir: The directory where the face images will be saved
        :param n: The number of pictures to capture
        """
        self.face_detector = face_detector
        self.cap = cv2.VideoCapture(0)  # Open the webcam
        self.cap.set(3, 640)  # Set frame width
        self.cap.set(4, 480)  # Set frame height
        self.save_dir = save_dir
        self.n = n  # Number of pictures to capture

    def run(self) -> Result[None]:
        """
        Starts capturing images of the user's face, saves them, and stops after capturing n images.
        """
        name = self._get_user_name()
        dir_path = self._create_user_directory(name)
        count = 0  # Counter to track the number of pictures taken

        while True:
            success, img = self._capture_frame()
            if not success:
                return Err("Could not read frame")

            # Display the live video feed in a window
            cv2.imshow("Face Registration - Press 'q' to quit", img)

            # Detect face and save if detected
            if self._capture_and_save_face(img, name, dir_path):
                count += 1
                print(f"Captured {count}/{self.n} face images")

            # Break the loop once the required number of faces are captured
            if count >= self.n:
                print("Successfully captured enough face images.")
                break

            # Quit on pressing 'q'
            if self._quit_requested():
                print("Quitting...")
                break

        self._cleanup()

        return Ok(None)

    def _get_user_name(self) -> str:
        """Prompt the user for their name."""
        return input("Enter your name: ")

    def _create_user_directory(self, name: str) -> str:
        """Create a directory for the user to save their images."""
        dir_path = os.path.join(self.save_dir, name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return dir_path

    def _capture_frame(self) -> Optional[tuple]:
        """Capture a frame from the camera."""
        success, img = self.cap.read()
        return success, img

    def _capture_and_save_face(self, img, name: str, dir_path: str) -> bool:
        """
        Detect a face, crop it, and save it with a timestamp if detected.
        
        Returns True if a face was detected and saved, False otherwise.
        """
        top_left, bottom_right = self.face_detector.get_bounding_box(img)
        if top_left and bottom_right:
            # Crop the face image from the frame
            face_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Save the face image with a timestamp
            self._save_face_image(face_img, name, dir_path)
            return True
        return False

    def _save_face_image(self, face_img, name: str, dir_path: str) -> None:
        """Save the captured face image with a timestamped filename."""
        now = time.time()
        timestamp = int(now * 1000)  # Convert time to milliseconds for better precision
        filename = f"{name}_{timestamp}.jpg"
        cv2.imwrite(os.path.join(dir_path, filename), face_img)

    def _quit_requested(self) -> bool:
        """Check if the 'q' key was pressed to quit the process."""
        return cv2.waitKey(1) == ord('q')

    def _cleanup(self) -> None:
        """Release the camera and close all OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()

        """
        Starts capturing images of the user’s face, saves them, and stops after capturing n images.

        :return: Ok(None) if successful, Err if an error occurs
        """
        name = input("Enter your name: ")
        dir_path = os.path.join(self.save_dir, name)

        # Ensure the directory exists, if not, create it
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        count = 0  # Counter to track the number of pictures taken

        while count < self.n:
            success, img = self.cap.read()
            if not success:
                return Err("Could not read frame")

            # Display the live video feed in a window
            cv2.imshow("Face Registration - Press 'q' to quit", img)

            top_left, bottom_right = self.face_detector.get_bounding_box(img)
            if top_left and bottom_right:
                # Crop and save the face image
                face_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                now = time.time()
                timestamp = int(now * 1000)
                filename = f"{name}_{timestamp}.jpg"
                cv2.imwrite(filename=filename, img=face_img)
                count += 1
                print(f"Captured {count}/{self.n} face images")

            if cv2.waitKey(1) == ord('q'):  # Quit on pressing 'q'
                print("Quitting...")
                break

        # Release the camera and close the window
        self.cap.release()
        cv2.destroyAllWindows()
        return Ok(None)
    

if __name__ == "__main__":
    
    face_detector = FaceDetector("yolo-Weights/yolo11-face-custom.pt")
    while True:
        choice = input("1. Register faces\n2. Recognize faces\n3. Exit\nEnter your choice: ")
        match choice:
            case "1":
                face_register = FaceRegister(face_detector=face_detector, save_dir="db", n=10)
                face_register.run()
            case "2":
                face_recognizer = FaceRecognizer(known_faces_dir="db", model=None)
                app = FaceRecognitionApp(face_detector, face_recognizer)
                app.run()
            case "3":
                print("Exiting...")
                quit()
            case _:
                print("Invalid choice")
                os.system('cls' if os.name == 'nt' else 'clear')


