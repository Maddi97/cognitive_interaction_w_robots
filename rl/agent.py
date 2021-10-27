from environment import Environment
from sys import path

path.append('control_unit')
from model.neural_network import Network
from CONSTANTS import *
import numpy as np


class Agent:
    def __init__(self):

        self.env = Environment(scan_interval=10)
        self.dqn = Network(len(ACTIONS), 17)

    def main_loop(self):
        while True:
            state = self.env.get_current_state()
            print(state)
            while True:
                action = self.act(state)

                # play song until end or thumbs down and collect reward
                reward = self.env.play_and_observe_negative()
                next_state = self.env.get_current_state()
                print(next_state)
                # calc something
                self.dqn.train(np.array([state]), np.array([[reward, 0, 0, 0]]))
                state = next_state

    # TODO
    def act(self, state):
        return np.random.choice(ACTIONS)

    def calc_q_value(self, state, action, reward):
        # state  input nn
        # estimated reward for every action ouput nn

        return np.argmax(self.dqn.predict(state))


if __name__ == '__main__':
    Agent().main_loop()
