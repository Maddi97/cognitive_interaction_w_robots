import numpy as np
from sys import path
path.append('rl')
from helpers import SONGS
from rl.model.neural_network import Network
from helpers import SONGS
import random
import tensorflow as tf

class Agent:
    def __init__(self, name, field_experiment=False, random = False, load=False):

        if not load:
            self.dqn = Network(len(SONGS), (17,), load=load, name = name)
        else:
            self.dqn = Network(len(SONGS), (17,), load=load, name=name)

        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.15  # minimum exploration probability
        self.epsilon_decay = 0.25
        self.field_experiment = field_experiment
        self.ran = random
        self.decay_step = 0

    def predict_song(self, state):

        self.decay_step += 1
        # EPSILON GREEDY STRATEGY
        if self.field_experiment:

            self.epsilon -= self.epsilon_decay
            explore_probability = self.epsilon
        # other strategy
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_probability = self.epsilon

        if (( explore_probability > np.random.rand() and self.epsilon>self.epsilon_min) or self.ran ):
            # Make a random action (exploration)
            #print("Random choice : " + str(explore_probability))
            return SONGS.index(np.random.choice(SONGS)), [], explore_probability, 'random'
        else:
            #print("Network choice : " + str(explore_probability))
            pred = self.dqn.predict(state)
            song = np.argmax(pred)
            return song, pred, explore_probability, 'network'

    def train(self, state, reward):
        history = self.dqn.train(np.array([state]), np.array([reward]))
        #print('training worked')
        return history, self.dqn.get_model()
