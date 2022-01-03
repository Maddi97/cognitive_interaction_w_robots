import numpy as np
import cv2
from pathlib import Path
SONGS = ['happy', 'positive', 'guitar', 'piano', 'commercial', 'upbeat', 'fun']

def mean_array(arr):
    return np.array(arr).mean(axis=0)

def max_array(arr):
    cnt = []
    for gest in arr:
        cnt.append(np.argmax(gest))
    pred = [0 for i in range(len(arr[0]))]
    pred[max(set(cnt), key=cnt.count)] = 1
    return np.array(pred)

def get_gesture(handTracker, state):
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    # emotion, em_pred = self.emDetection.predict_emotion(img)
    img = handTracker.track_hand(img)
    gesture, gesture_pred = handTracker.predict_gesture(img)
    visualize_prediction(img, state,  gesture)
    cap.release()
    cv2.destroyAllWindows()
    for i in range(5):  # maybe 5 or more
        cv2.waitKey(1)
    return gesture

def visualize_prediction(img, state, gesture):
    cv2.startWindowThread()

    cv2.putText(img, state, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # set your desired size
    cv2.resizeWindow('Image', 1400, 900)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('quit')

