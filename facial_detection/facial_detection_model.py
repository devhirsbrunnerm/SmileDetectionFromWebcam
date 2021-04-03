import cv2
import base64
import numpy as np
import os


class FacialDetectionModel:
    def __init__(self):
        self.face_cascade = None
        self.__load_cascade_classifier()

    def detect(self, image):
        # convert the image to grayscale (necessary for face detection)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # find all faces on the image
        faces = self.face_cascade.detectMultiScale(image)
        # collect the cropped images of the faces as well as their coordinates in the original
        face_images = []
        for (x, y, w, h) in faces:
            # crops the original image to only the face
            face_image = image[y:y+h, x:x+w]
            face_images.append(((x, y, w, h), face_image))
        
        return face_images



    def __load_cascade_classifier(self):
        print('Loading Facial detection with haarcascade_frontalface_default.xml')
        dirname = os.path.dirname(__file__) 
        self.face_cascade = cv2.CascadeClassifier(f'{dirname}/haarcascade_frontalface_default.xml')
