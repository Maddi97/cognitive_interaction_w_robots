from sys import path
import numpy as np

import cv2

from tracking.tracking_algorithms import emotion_detection
from tracking.tracking_algorithms.bodyTracker import BodyTracker
from tracking.tracking_algorithms.handTracker import HandTracker

path.append('tracking/tracking_algorithms')  # for module import


class Environment:

    def __init__(self, scan_interval=1000):
        self.handTracker = HandTracker(mode=True)
        self.bodyTracker = BodyTracker()
        self.emDetection = emotion_detection.EmotionDetection()

        self.scan_interval = scan_interval
        self.actions = ['slow', 'fast', 'loud', 'trash']

    def get_current_state(self):
        cap = cv2.VideoCapture(0)
        for i in range(self.scan_interval):
            success, img = cap.read()
            emotion, em_pred = self.emDetection.predict_emotion(img)
            img = self.handTracker.track_hand(img)
            gesture, gesture_pred = self.handTracker.predict_gesture(img)

            state = np.concatenate((em_pred.squeeze(), gesture_pred.squeeze()))

            cv2.putText(img, gesture, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, emotion, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return state

    def act(self):
        return np.random.choice(self.actions)

    def observing_negative(self):
        while (True):
            cap = cv2.VideoCapture(0)
            success, img = cap.read()
            emotion, em_pred = self.emDetection.predict_emotion(img)
            img = self.handTracker.track_hand(img)
            gesture, gesture_pred = self.handTracker.predict_gesture(img)

            state = np.concatenate((em_pred.squeeze(), gesture_pred.squeeze()))
            print(state)
