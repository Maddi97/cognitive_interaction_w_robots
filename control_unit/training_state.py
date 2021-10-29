import numpy as np


class TrainingState(object):
    def __init__(self, agent):
        self.agent = agent

    def train(self, state, reward):
        # TODO
        print('traaaain')
        state = np.reshape(state, (17))
        self.agent.train(state=state, reward=reward)
