import cv2


class QuestioningState(object):
    def __init__(self, handTracker, scan_interval=100, stopping_crit=5):
        self.handTracker = handTracker
        self.scan_interval = scan_interval
        self.stopping_crit = stopping_crit

    def decideForState(self):
        count_down = 0
        count_up = 0
        stop = 0

        while 1:
            #TODO
            return
            gesture = self.__get_gesture__()
            if gesture == 'thumbs down':
                count_down += 1

            elif gesture == 'thumbs up':
                count_up += 1

            elif gesture == 'stop':
                stop += 1

            if count_down == self.stopping_crit:
                return 'thumbs down'

            elif count_up == self.stopping_crit:
                return 'thumbs up'

            elif stop == self.stopping_crit:
                return 'stop'


    def __get_gesture__(self):

        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        img = self.handTracker.track_hand(img)
        gesture, gesture_pred = self.handTracker.predict_gesture(img)
        self.__visualize_prediction__(img, gesture=gesture)

        return gesture


    def __visualize_prediction__(self, img, gesture=None, emotion=None):
        cv2.putText(img, 'Questioning State', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, emotion, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
