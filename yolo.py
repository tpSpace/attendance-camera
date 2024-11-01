from os import listdir
import numpy as np
from ultralytics import YOLO
import cv2
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import math

from compare import loadVggFaceModel, findCosineSimilarity
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
# model = YOLO("yolo-Weights/yolov8n.pt")
yolo_model = YOLO("yolo-Weights/yolov8n.pt")



# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

vgg_model = loadVggFaceModel()

people_pictures = "./db"

people = dict()
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    return img
for file in listdir(people_pictures):
	person, extension = file.split(".")
	people[person] = vgg_model.predict(preprocess_image('./db/%s.jpg' % (person)))[0,:]


while True:
    success, img = cap.read()
    results = yolo_model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
           

            # class name
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
                print("Class name -->", classNames[cls])
                detected_face = img[int(y1):int(y2), int(x1):int(x2)] 
                detected_face = cv2.resize(detected_face, (224, 224))
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 127.5
                img_pixels -= 1
                captured_representation = vgg_model.predict(img_pixels)[0,:]
                found = 0
                for person in people:
                    person_name = person
                    representation = people[person]
                    
                    similarity = findCosineSimilarity(representation, captured_representation)
                    if(similarity < 0.20):
                        print("Detect", person_name)
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, person_name, org, font, fontScale, color, thickness)
                        found = 1
                        break
                    

                # object details
                

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))