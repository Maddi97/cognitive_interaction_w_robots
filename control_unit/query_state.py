import time

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
        states_em = []
        states_gest = []
        for i in range(self.scan_interval):
            success, img = cam.read()
            state_em, state_gest, gesture, emotion = self.__get_current_state__(img)
            self.__visualize_prediction__(img, gesture, emotion)

            states_em.append(state_em)
            states_gest.append(state_gest)
        cam.release()
        cv2.destroyAllWindows()
        for i in range(5):  # maybe 5 or more
            cv2.waitKey(1)

        mean_emotion_prediction = helpers.mean_array(states_em)
        max_gesture_prediction = helpers.max_array(states_gest)

        final_prediction = np.reshape(np.concatenate((mean_emotion_prediction, max_gesture_prediction)), (1,17))

        return final_prediction

    def __get_current_state__(self, img):
        emotion, em_pred = self.emDetection.predict_emotion(img)
        img = self.handTracker.track_hand(img)
        gesture, gesture_pred = self.handTracker.predict_gesture(img)
        pred = [0 for i in range(len(gesture_pred[0]))]

        pred[np.argmax(gesture_pred)] = 1
        #print(pred)
        #state = np.concatenate((em_pred.squeeze(), np.array(pred).squeeze()))
        return em_pred.squeeze(),np.array(pred).squeeze() , gesture, emotion

    def __visualize_prediction__(self, img, gesture=None, emotion=None):
        cv2.startWindowThread()

        cv2.putText(img, 'Query State', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, emotion, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')