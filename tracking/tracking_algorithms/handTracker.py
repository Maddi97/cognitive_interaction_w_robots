import cv2
import mediapipe as mp
import numpy as np
from ..CONSTANTS import HAND_LANDMARKS
from tensorflow.keras.models import load_model
from pathlib import Path


class HandTracker():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.model = load_model(str(Path(__file__).parent.parent) + '/tracking_algorithms/model/mp_hand_gesture')

        # Load class names
        f = open(str(Path(__file__).parent.parent) + '/tracking_algorithms/gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()

    def track_hand(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        landmarks = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    break
                    #self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def get_landmark_data(self, img, handNo=0):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_obj = {'id': HAND_LANDMARKS[id], 'x': lm.x, 'y': lm.y, 'z': lm.z, 'cx': cx, 'cy': cy}
                landmarks.append(landmark_obj)
                # print(landmark_obj)
        return landmarks

    def predict_gesture(self, img, handNo=0):
        landmarks = []
        prediction = np.zeros((1, 10))
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
                # print(landmark_obj)

                # Predict gesture in Hand Gesture Recognition project
            prediction = self.model.predict([landmarks])

            classID = np.argmax(prediction)
            className = self.classNames[classID]

            return className, prediction

        else:
            return 'Nothing', prediction
