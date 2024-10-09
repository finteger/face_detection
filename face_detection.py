import face_recognition
import cv2
import numpy as np

#Get a reference to the default webcam
video_capture = cv2.VideoCapture(0)

#Load a sample picture and learn how to recognize it
your_image = face_recognition.load_image_file("todd.jpg")
your_face_encoding = face_recognition.face_encodings(your_image)[0]

#Create an array of known face encodings
known_face_encodings = [
    your_face_encoding,
]