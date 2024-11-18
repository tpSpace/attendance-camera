import logging
import numpy as np
import cv2
from typing import Any, List, Tuple
from .detection import FaceDetector
from .alignment import FaceAligner
from .model import BoundingBox, Person

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Pipeline:
    """
    A pipeline for face detection in a camera stream.
    """
    STANDARD_DIMENSION = (244, 244)
    def __init__(self, conf_threshold: float = 0.5):
        self.detector = FaceDetector(conf_threshold=conf_threshold)
        self.aligner = FaceAligner(desiredFaceWidth=256)
        logging.info("Pipeline initialized with confidence threshold: %f", conf_threshold)
        

    def stream(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Could not open camera.")
            return
        logging.info("Camera stream session started!")
        people: List[Person] = []
        while True:
            capture: Tuple[Any, np.ndarray] = cap.read()
            ret, frame = capture
            if not ret:
                logging.error("Could not read frame.")
                break
            boxes = self.detector.detect_faces(frame)
            for box in boxes:
                if box is not None:
                    person = Person()
                    self.aligner.preprocess(image=frame, box=box)
                    aligned = self.aligner.align()
                    people.append(person)
                    cv2.imshow("Aligned", aligned)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("Exiting camera stream.")
                        break
            
            # self._draw(frame, boxes)
            
            # cv2.imshow("frame", frame)

           
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera stream session closed!")

    def from_image(self, path: str) -> None:
        image = cv2.imread(path)
        boxes = self.detector.detect_faces(image)
        people: List[Person] = []
        for box in boxes:
            if box is not None:
                person = Person()
                people.append(person)
                self.aligner.preprocess(image=image, box=box)
                aligned = self.aligner.align()
                while True:
                    cv2.imshow("Aligned", aligned)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("Exiting camera stream.")
                        break
    
    def _draw(self, frame: np.ndarray, boxes: List[BoundingBox]) -> None:
        for i, box in enumerate(boxes):
            landmarks = self.aligner.get_landmarks_from_image(image=frame, detected_face=box)
            print(type(landmarks))
            print(landmarks)
            x1, y1, x2, y2, _ = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(frame, f"Confidence: {box.confidence:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # Draw the landmarks
            if landmarks is not None:
                for pred in landmarks:
                    for (x, y, z) in pred:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        
   