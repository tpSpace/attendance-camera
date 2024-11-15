import os
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
from .model import BoundingBox



class FaceDetector:
    def __init__(self, conf_threshold: float = 0.5):
        """
        Initializes the YOLO face detector.
        """
        model_path = os.path.join(os.path.dirname(__file__), "../pretrain/yolo11-face-custom.pt")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, img: np.ndarray) -> List[BoundingBox]:
        """
        Detect faces in an image using YOLO.
        Returns a list of tuples with bounding boxes for detected faces.
        Each bounding box is represented as a tuple: (x1, y1, x2, y2, confidence).
        """
        results = self.model(img)
        detected_faces = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0 and box.conf[0] >= self.conf_threshold:
                    x1, y1, x2, y2, confidence = self._process_face(box)
                    detected_faces.append(BoundingBox(x1, y1, x2, y2, confidence))
        return detected_faces
    
    def crop(self, image: np.ndarray, box: BoundingBox) -> np.ndarray:
        """
        Crop the image to the bounding box.
        """

        return image[box.y1:box.y2, box.x1:box.x2]

    def _process_face(self, box) -> Tuple[int, int, int, int, float]:
        """
        
        """
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = box.conf[0].item()
        return x1, y1, x2, y2, confidence
    
