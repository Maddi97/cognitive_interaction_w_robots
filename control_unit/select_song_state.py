from sys import path
from helpers import SONGS


class SongSelectionState(object):
    def __init__(self, agent):
        self.agent = agent

    def select_song(self, state):
        song = SONGS[self.agent.predict_song(state)]
        # TODO
        # choose a song from selected genre
        return song

