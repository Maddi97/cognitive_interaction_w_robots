import numpy as np
from sys import path
path.append('rl')

from rl.model.neural_network import Network
from rl. CONSTANTS import *


class Agent:
    def __init__(self):
        self.dqn = Network(len(ACTIONS), (17,))

    # TODO
    def predict_song(self, state):
        pred = self.dqn.predict(state)
        print(pred)
        song = np.argmax(pred)

        # return ACTIONS.index(np.random.choice(ACTIONS))
        return song

    def train(self, state, reward):
        self.dqn.train(np.array([state]), np.array([reward]))


if __name__ == '__main__':
    Agent().main_loop()
