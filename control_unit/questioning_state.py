import cv2
from helpers import *
from pygame import mixer
import time
class QuestioningState(object):
    def __init__(self, handTracker, scan_interval=100, stopping_crit=5):
        self.handTracker = handTracker
        self.scan_interval = scan_interval
        self.stopping_crit = stopping_crit
        self.state = 'questioning state'

    def decideForState(self):

        mixer.init()
        mixer.music.load('../assets/audio/questioning state A.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(1)

        count_down = 0
        count_up = 0
        count_stop = 0
        while 1:  # wait for music to finish playing
            gesture = get_gesture(self.handTracker, self.state)
            if gesture == 'thumbs down':
                count_down += 1
            elif gesture == 'thumbs up':
                count_up += 1

            elif gesture == 'stop':
                count_stop += 1

            if count_down == self.stopping_crit:
                print('abbruch')
                return 'up'

            if count_up == self.stopping_crit:
                print('human found and willing to play')
                return 'down'

            if count_stop == self.stopping_crit:
                return 'stop'

