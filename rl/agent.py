import numpy as np
from sys import path
path.append('rl')

from rl.model.neural_network import Network
from rl. CONSTANTS import *


class Agent:
    def __init__(self):

        self.dqn = Network(len(ACTIONS), 17)

    # TODO
    def predict_song(self, state):
        # song = np.argmax(self.dqn.predict(state))

        return ACTIONS.index(np.random.choice(ACTIONS))

    def train(self, state, reward):
        self.dqn.train(np.array([state]), np.array([reward]))


if __name__ == '__main__':
    Agent().main_loop()
