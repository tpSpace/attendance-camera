from deepface import DeepFace

# anti spoofing test in face detection
face_objs = DeepFace.extract_faces(
  img_path="db/khoi.jpg",
  anti_spoofing = True
)

# anti spoofing test in real time analysis
DeepFace.stream(
    db_path = "db",
    anti_spoofing = True,
    frame_threshold=2,
    time_threshold=2,
    enable_face_analysis=False,
    # model_name="yolov8"
)