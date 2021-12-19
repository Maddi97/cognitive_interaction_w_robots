from helpers import SONGS


class SongSelectionState(object):
    def __init__(self, agent):
        self.agent = agent

    def select_song(self, state):
        song, pred, epsilon, choice = self.agent.predict_song(state)
        song = SONGS[song]
        # TODO
        # choose a song from selected genre
        return song, pred, epsilon, choice

