import numpy as np

import cv2


import helpers

class QueryState(object):
    def __init__(self, scan_interval = 100, handTracker=None, bodyTracker=None, emDetection=None):
        self.handTracker = handTracker
        self.bodyTracker = bodyTracker
        self.emDetection = emDetection

        self.scan_interval = scan_interval


    def scan_human(self):
        cam = cv2.VideoCapture(0)
        states = []
        for i in range(self.scan_interval):
            success, img = cam.read()
            state, gesture, emotion = self.__get_current_state__(img)

            self.__visualize_prediction__(img, gesture, emotion)
            states.append(state)

        return helpers.mean_array(states)

    def __get_current_state__(self, img):
        emotion, em_pred = self.emDetection.predict_emotion(img)
        img = self.handTracker.track_hand(img)
        gesture, gesture_pred = self.handTracker.predict_gesture(img)
        state = np.concatenate((em_pred.squeeze(), gesture_pred.squeeze()))
        return state, gesture, emotion

    def __visualize_prediction__(self, img, gesture=None, emotion=None):
        cv2.putText(img, 'Query State', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, emotion, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')