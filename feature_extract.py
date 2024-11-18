import numpy as np
import dlib
import cv2
from PIL import Image
from deepface import DeepFace

img_path = "/content/drive/MyDrive/Metaverse ML/Images/sd.png"

# Reading the image and converting it into an numpy array

img = dlib.load_rgb_image(img_path) 

# Gender and Race Detection

res = DeepFace.analyze(img, actions = ['gender','race'])

print(f"{res['gender']}")
print(f"{res['dominant_race']}")

# Glasses Detection

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(r"/content/drive/MyDrive/Metaverse ML/Dependencies/shape_predictor_68_face_landmarks.dat")     # pre-trained model for 68 face landmark prediction

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

nose_bridge_x = []
nose_bridge_y = []
for i in [28,29,30,31,33,34,35]:
        nose_bridge_x.append(landmarks[i][0])
        nose_bridge_y.append(landmarks[i][1])
        
x_min = min(nose_bridge_x)
x_max = max(nose_bridge_x)

y_min = landmarks[20][1]
y_max = landmarks[31][1]
img2 = Image.open(img_path)
img2 = img2.crop((x_min,y_min,x_max,y_max))

img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
edges = cv2.Canny(image =img_blur, threshold1=85, threshold2=120)

edges_center = edges.T[(int(len(edges.T)/2))]

if 255 in edges_center:
 print('Glasses are present')
else:
 print('Glasses are absent')


# Beard Detection

face_cascade = cv2.CascadeClassifier(r"/content/drive/MyDrive/Metaverse ML/Dependencies/haarcascade_frontalface_default.xml")      # Pre-trained model for face detection

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1,5)

for (x,y,w,h) in faces:
  mask = np.zeros_like(img)
  mask = cv2.ellipse(mask, (int((x+w)/1.2), y+h),(69,69), 0, 0, -180, (255,255,255),thickness=-1)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

  result = np.bitwise_and(img, mask)
  hsv_img = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
  low_black = np.array([94, 80, 2])
  high_black = np.array([126, 255, 255])
  MASK = cv2.inRange(hsv_img, low_black, high_black)

  if cv2.countNonZero(MASK) == 0:
    print("Beard Not Found")
  else:
    print("Beard Found")
     
img_path ="/content/drive/MyDrive/Images/sd.png"

img = cv2.imread(img_path)
res=DeepFace.analyze(img, actions = ['race','gender'])
print(f"{res['dominant_race']}")
print(f"{res['gender']}")
     