import cv2
from pygame import mixer
import time
class QuestioningState(object):
    def __init__(self, handTracker, scan_interval=100, stopping_crit=5):
        self.handTracker = handTracker
        self.scan_interval = scan_interval
        self.stopping_crit = stopping_crit
        self.state = 'questioning state'

    def decideForState(self):

        count_down = 0
        count_up = 0
        count_stop = 0
        cam = cv2.VideoCapture(0)

        while 1:  # wait for music to finish playing
            gesture = get_gesture(self.handTracker, self.state,cam)
            if gesture == 'thumbs down':
                count_down += 1
            elif gesture == 'thumbs up':
                count_up += 1

            elif gesture == 'stop':
                count_stop += 1

            if count_down == self.stopping_crit:
                print('abbruch')
                cam.release()
                cv2.destroyAllWindows()
                for i in range(5):  # maybe 5 or more
                    cv2.waitKey(1)
                return 'down'

            if count_up == self.stopping_crit:
                print('human found and willing to play')
                cam.release()
                cv2.destroyAllWindows()
                for i in range(5):  # maybe 5 or more
                    cv2.waitKey(1)
                return 'up'

            if count_stop == self.stopping_crit:
                cam.release()
                cv2.destroyAllWindows()
                for i in range(5):  # maybe 5 or more
                    cv2.waitKey(1)
                return 'stop'

def get_gesture(handTracker, state, cam):
    success, img = cam.read()
    # emotion, em_pred = self.emDetection.predict_emotion(img)
    img = handTracker.track_hand(img)
    gesture, gesture_pred = handTracker.predict_gesture(img)
    visualize_prediction(img, state,  gesture)

    for i in range(5):  # maybe 5 or more
        cv2.waitKey(1)
    return gesture

def visualize_prediction(img, state, gesture):
    cv2.startWindowThread()

    cv2.putText(img, state, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('quit')