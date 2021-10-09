import cv2
import mediapipe as mp
import time


class HandTracker():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def track_hand(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def get_landmark_data(self, img, handNo=0):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_obj = {'id': id, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'cx': cx, 'cy': cy}
                landmarks.append(landmark_obj)
                # print(landmark_obj)
        return landmarks
