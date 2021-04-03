import os
# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import cv2
from model.smile_detection_model import SmileDetectionModel
from facial_detection.facial_detection_model import FacialDetectionModel

print('Setting up...')
cap = cv2.VideoCapture(0)
fdm = FacialDetectionModel()
sdm = SmileDetectionModel()

record = cv2.VideoWriter(
    'smile_detection.avi', 
    cv2.VideoWriter_fourcc(*'XVID'),
    20.0,
    (640, 480))

print('Ready to use...')

# As long as the application is running
while(True):
    # The capture can fail at some points for single frames, this is not important 
    # but will crash the code without wrapping it into try and except
    try:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Detect all faces in the frame
        faces = fdm.detect(frame)

        # For each face, detect wheter it is smiling or not and draw a border and positive or negative
        for (x, y, w, h), face_image in faces:
            # draws border around face
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

            # gets text wheter smiling is positive or negative
            smile = sdm.predict(face_image)

            # writes the text in the bottom left of the rectangle that shows the face
            cv2.putText(frame, smile, (x, y + h + 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        record.write(frame)
    except Exception as e:
        print(e)
    # when pressing q in the application quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
record.release()
cv2.destroyAllWindows()
