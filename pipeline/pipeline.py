import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1

# ... [other imports]
import logging
import numpy as np
import cv2
from typing import Any, List, Tuple
from .detection import FaceDetector
from .alignment import FaceAligner
from .model import BoundingBox, Person
class Pipeline:
    STANDARD_DIMENSION = (244, 244)

    def __init__(self, conf_threshold: float = 0.5):
        self.detector = FaceDetector(conf_threshold=conf_threshold)
        self.aligner = FaceAligner(desiredFaceWidth=256)
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.embeddings_db = self.load_embeddings()
        logging.info("Pipeline initialized with confidence threshold: %f", conf_threshold)

    def load_embeddings(self):
        embeddings = []
        names = []
        db_path = 'db'
        for filename in os.listdir(db_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(db_path, filename)
                img = Image.open(img_path).convert('RGB')
                face_tensor = self.preprocess_image(img)
                with torch.no_grad():
                    embedding = self.embedding_model(face_tensor.unsqueeze(0))
                embeddings.append(embedding.numpy())
                names.append(os.path.splitext(filename)[0])
        return {'embeddings': embeddings, 'names': names}

    def preprocess_image(self, img):
        img = img.resize((160, 160))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        img_tensor /= 255.0  # Normalize to [0,1]
        return img_tensor

    def stream(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Could not open camera.")
            return
        logging.info("Camera stream session started!")
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Could not read frame.")
                break
            boxes = self.detector.detect_faces(frame)
            for box in boxes:
                if box is not None:
                    landmark = self.aligner.get_landmarks_from_image(image=frame, detected_face=box)
                    face = box.crop(image=frame)
                    face = cv2.resize(face, self.STANDARD_DIMENSION)
                    aligned = self.aligner.align(landmark=landmark, face=face)
                    
                    # Convert aligned face to tensor
                    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(aligned_rgb)
                    face_tensor = self.preprocess_image(img)
                    
                    # Get embedding of the aligned face
                    with torch.no_grad():
                        embedding = self.embedding_model(face_tensor.unsqueeze(0))
                    
                    # Compare with embeddings in the database
                    similarities = []
                    for db_emb in self.embeddings_db['embeddings']:
                        similarity = cosine_similarity(embedding.numpy(), db_emb)
                        similarities.append(similarity[0][0])
                    if similarities:
                        best_match_index = np.argmax(similarities)
                        best_similarity = similarities[best_match_index]
                        threshold = 0.8  # Adjust threshold as needed
                        if best_similarity > threshold:
                            name = self.embeddings_db['names'][best_match_index]
                        else:
                            name = "Unknown"
                        # Draw the name and similarity on the frame
                        x1, y1, x2, y2, _ = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({best_similarity:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow("Aligned", aligned)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting camera stream.")
                break
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera stream session closed!")