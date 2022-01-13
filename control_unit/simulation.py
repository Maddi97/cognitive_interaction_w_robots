
from control_unit.query_state import QueryState
from control_unit.init_state import InitState
from control_unit.play_song_state import PlaySongState
from control_unit.questioning_state import QuestioningState
from control_unit.select_song_state import SongSelectionState
from control_unit.training_state import TrainingState

from tracking.tracking_algorithms import emotion_detection
from tracking.tracking_algorithms.bodyTracker import BodyTracker
from tracking.tracking_algorithms.handTracker import HandTracker

from rl.agent import Agent
from helpers import SONGS


import random
import numpy as np
import pandas as pd

simulation_song = 'happy'

handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker()
emDetection = emotion_detection.EmotionDetection()

agent = Agent(field_experiment=False, random=False, load=False, name=simulation_song)

select_song_state = SongSelectionState(agent)

training_state = TrainingState(agent=agent)

#actions = ['happy', 'positive', 'guitar', 'piano', 'commercial', 'upbeat', 'fun']
actions = ['happy']
counter = {'happy': 0, 'positive': 0, 'guitar': 0, 'piano': 0, 'commercial': 0, 'upbeat': 0, 'fun': 0}

df = pd.DataFrame([], columns = ['Iteration', 'Epsilon', 'Choice', 'Song', 'Reward', 'Q-Values', 'MSE'])

model = None


for iteration in range(3000):
    #print('Iteration: {}'.format(iteration))
    state_em = np.random.rand(7)
    state_gest = [0 for i in range(10)]
    i = random.randint(0, 9)

    state_gest[i] = 1

    state = np.reshape(np.concatenate((state_em, state_gest)), (1,17))
    song, pred, epsilon, choice = select_song_state.select_song(state)
    counter[song] += 1
    reward = np.zeros(shape=(len(SONGS)))
    index = SONGS.index(song)
    r = 0

    if song in actions:
        r = 10
        reward[index] = 10
    else:
        r = -10
        reward[index] = -10


    history, dqn = training_state.train(state, reward)

    model = dqn
    #print("#####" + song + "######\n", reward)
    #print(counter)
    #print(history['mse'])
    df_row = pd.Series({'Iteration': iteration,'Epsilon':epsilon, 'Choice':choice, 'Song': song, 'Reward': r, 'Q-Values':pred, 'MSE':history['mse'][0]})
    df = df.append(df_row, ignore_index=True)

model.save("../results/simulation/{}_sim_model".format(simulation_song))
#df.to_csv("../results/{}_results.csv".format(simulation_song))