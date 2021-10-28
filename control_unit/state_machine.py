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

import time

handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker()
emDetection = emotion_detection.EmotionDetection()

agent = Agent()

init_state = InitState()
query_state = QueryState(scan_interval=10, handTracker=handTracker, emDetection=emDetection)
play_song_state = PlaySongState(handTracker=handTracker)
questioning_state = QuestioningState(handTracker=handTracker)
select_song_state = SongSelectionState(agent)
training_state = TrainingState(agent=Agent)



init_state.find_human()

while 1:
    state = query_state.scan_human()
    time.sleep(2)
    song = select_song_state.select_song(state)
    time.sleep(2)
    reward = play_song_state.play_song(song)
    time.sleep(2)
    training_state.train(state, reward)
    time.sleep(2)
    questioning_state.decideForState()
    time.sleep(2)

