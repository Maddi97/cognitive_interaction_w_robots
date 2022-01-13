from helpers import *
import time

class InitState(object):
    def __init__(self, stopping_crit, scan_interval,  handTracker, emDetection):
        self.stopping_crit = stopping_crit
        self.handTracker = handTracker
        self. emDetection = emDetection
        self.scan_interval = scan_interval
        self.state = 'init state'
        self.found = True

    def find_human(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        # set your desired size
        cv2.resizeWindow('Window', 1400, 900)
        count = 0
        timeout = time.time() + 60 * self.scan_interval
        while 1:
            success, img = cam.read()
            self.__visualize_prediction__(img)
            face = self.emDetection.get_faces(img)
            if len(face) == 0:
                print("no face")
            else:
                count += 1

            if count >= self.stopping_crit:
                #print("Human found")
                cam.release()
                cv2.destroyAllWindows()
                for i in range(5):  # maybe 5 or more
                    cv2.waitKey(1)
                return True

            if time.time() > timeout:
                return False


    def __visualize_prediction__(self, img, gesture=None, emotion=None):
        cv2.startWindowThread()

        cv2.putText(img, 'Searching for a human ...', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, emotion, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # set your desired size
        cv2.resizeWindow('Image', 1400, 900)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
