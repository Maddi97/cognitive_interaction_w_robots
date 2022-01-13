import numpy as np


class TrainingState(object):
    def __init__(self, agent):
        self.agent = agent

    def train(self, state, reward):
        state = np.reshape(state, (17))
        history, dqn = self.agent.train(state=state, reward=reward)
        return history, dqn


# [0.12469387 0.14992487 0.16767557 0.01381204 0.26916435 0.26140749
#  0.0133218  1.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.        ]

# [0.91394132 0.41410852 0.6835003  0.26550186 0.45170426 0.0785872
#  0.74663972 0.         0.         0.         0.         0.
#  0.         0.         1.         0.         0.        ]