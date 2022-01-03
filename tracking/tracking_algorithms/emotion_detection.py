import cv2
import numpy as np
from tracking.tracking_algorithms.model.neural_network import Network
from pathlib import Path

class EmotionDetection:
    def __init__(self):
        self.model = Network().get_model()
        self.EMOTIONS_LIST = ["Angry", "Disgust",
                         "Fear", "Happy",
                         "Neutral", "Sad",
                         "Surprise"]

    def predict_emotion(self, img):
        # start the webcam feed
        facecasc = cv2.CascadeClassifier(str(Path(__file__).parent.parent) +
                                         '/tracking_algorithms/model/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        maxindex = 0
        prediction = np.zeros((1,7))
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

        return self.EMOTIONS_LIST[maxindex],  prediction

    def get_faces(self, img):
        facecasc = cv2.CascadeClassifier(str(Path(__file__).parent.parent) +
                                         '/tracking_algorithms/model/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces