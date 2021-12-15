import cv2
import numpy as np
from helpers import SONGS

import pandas as pd
import random
from pygame import mixer
import time

class PlaySongState(object):
    def __init__(self, stopping_crit=5, handTracker=None):
        print("Init music box")
        self.stopping_crit = stopping_crit
        self.handTracker = handTracker

        self.music_df = pd.read_csv('../assets/Musik/csv/music_tags_reduced.csv')

        self.music_path = '../assets/Musik/'

    def play_song(self, song):
        # play song
        mixer.init()
        mixer.music.load('../assets/audio/play song state A.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(1)

        reward = np.zeros(shape=(len(SONGS)))
        index = SONGS.index(song)
        count_down = 0
        count_up = 0

        songs = [''.join([self.music_path, 'bensound-', track.replace(' ', ''), '.mp3']) for track in
                 list(self.music_df.loc[self.music_df[song]]['track_name'])]
        mixer.music.load(random.sample(songs, 1).pop())
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            gesture = self.__get_gesture()
            if gesture == 'thumbs down':
                count_down += 1
            elif gesture == 'thumbs up':
                count_up += 1

            if count_down == self.stopping_crit:
                reward[index] = -10
                return reward

            if count_up == self.stopping_crit:
                reward[index] = 10
                return reward

    def __get_gesture(self):
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        # emotion, em_pred = self.emDetection.predict_emotion(img)
        img = self.handTracker.track_hand(img)
        gesture, gesture_pred = self.handTracker.predict_gesture(img)
        self.__visualize_prediction__(img, gesture)
        cap.release()
        cv2.destroyAllWindows()
        return gesture

    def __visualize_prediction__(self, img, gesture=None):
        cv2.putText(img, 'Play Song State', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, gesture, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')