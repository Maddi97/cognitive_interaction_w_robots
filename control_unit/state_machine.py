from query_state import QueryState
from init_state import InitState
from play_song_state import PlaySongState
from questioning_state import QuestioningState
from select_song_state import SongSelectionState
from training_state import TrainingState

from tracking.tracking_algorithms import emotion_detection
from tracking.tracking_algorithms.bodyTracker import BodyTracker
from tracking.tracking_algorithms.handTracker import HandTracker

from rl.agent import Agent
import numpy as np
import time


handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker()
emDetection = emotion_detection.EmotionDetection()

agent = Agent()

init_state = InitState(stopping_crit=1, scan_interval=1000, handTracker=handTracker, emDetection=emDetection)

query_state = QueryState(scan_interval=20, handTracker=handTracker, emDetection=emDetection)

play_song_state = PlaySongState(handTracker=handTracker, stopping_crit=1)

questioning_state = QuestioningState(handTracker=handTracker, stopping_crit=2)
select_song_state = SongSelectionState(agent)
training_state = TrainingState(agent=agent)


found = init_state.find_human()
skip_scan = False
state = []
while 1:

    if not found:
        print("Abort no human found")
        break
    time.sleep(2)
    if not skip_scan:
        state = query_state.scan_human()
        time.sleep(2)

    song = select_song_state.select_song(state)
    if not skip_scan:
        reward = play_song_state.play_song(song)
        training_state.train(state, reward)

    answer = questioning_state.decideForState()

    if answer == 'down':
        print('Okay another round')
        skip_scan = False
    elif answer == 'up':
        print('Okay here comes another song to your mood')
        skip_scan = True
    elif answer == 'stop':
        print('Okay Goodbye! I am happy to see u again!')
        break