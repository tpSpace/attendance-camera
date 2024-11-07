from deepface import DeepFace
import cv2

# List of available backends, models, and distance metrics
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

def realtime_face_recognition():
    model_name = models[0]
    model = DeepFace.build_model(model_name)
    
    # Define a video capture object
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

     
        
        # Perform face recognition on the captured frame
        people = DeepFace.find(img_path=frame, db_path="db/", model_name=model_name, distance_metric=metrics[2], enforce_detection=False)

        for person in people:
            # Check if the coordinates of the face bounding box exist and have at least one element
            if ('source_x' in person and len(person['source_x']) > 0 and
                'source_y' in person and len(person['source_y']) > 0 and
                'source_w' in person and len(person['source_w']) > 0 and
                'source_h' in person and len(person['source_h']) > 0):
                
                # Retrieve the coordinates of the face bounding box
                x = person['source_x'][0]
                y = person['source_y'][0]
                w = person['source_w'][0]
                h = person['source_h'][0]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get the person's name and display it on the image
                name = person['identity'][0].split('/')[1]
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                continue

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()

realtime_face_recognition()