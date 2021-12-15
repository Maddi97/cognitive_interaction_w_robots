from helpers import *


class InitState(object):
    def __init__(self, stopping_crit, scan_interval,  handTracker, emDetection):
        self.stopping_crit = stopping_crit
        self.handTracker = handTracker
        self. emDetection = emDetection
        self.scan_interval = scan_interval
        self.state = 'init state'

    def find_human(self):
        cam = cv2.VideoCapture(0)
        count = 0
        rounds = 0
        while 1:
            success, img = cam.read()

            face = self.emDetection.get_faces(img)
            if len(face) == 0:
                print("no face")
            else:
                count += 1

            if count >= self.stopping_crit:
                print("Human found")
                cam.release()
                cv2.destroyAllWindows()
                return True
            if rounds >= self.scan_interval:
                print("No human found")
                cam.release()
                cv2.destroyAllWindows()
                return False


