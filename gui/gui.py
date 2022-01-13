
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

import PySimpleGUI as sg
import sys
from pygame import mixer


NAME = "max_test"
LOAD = False


handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker()
emDetection = emotion_detection.EmotionDetection()

# define if simulation or field experiment and control group or not
agent = Agent(field_experiment=True, random=False, load=LOAD, name=NAME)

init_state = InitState(stopping_crit=10, scan_interval=1, handTracker=handTracker, emDetection=emDetection)

query_state = QueryState(scan_interval=20, handTracker=handTracker, emDetection=emDetection)

play_song_state = PlaySongState(handTracker=handTracker, stopping_crit=10)

questioning_state = QuestioningState(handTracker=handTracker, stopping_crit=5)
select_song_state = SongSelectionState(agent)
training_state = TrainingState(agent=agent)



class gui():
    def __init__(self):
        self.round = 0
        self.window = sg.Window("Jukebox", layout=[[sg.Text("Hello I am your JukeBot. Please Press any Button to continue", justification='center',size=(1200,5), font = ("Courier New", 20, "bold"), pad=(0,30))],
                                                   [sg.Image('../assets/smileys/smile.png', size=(500, 500))],
                                                   [sg.Button("Start",font = ("Courier New", 30, "bold"), pad=(10,0)), sg.Button("Tutorial",font = ("Courier New", 30, "bold"), pad=(20,0)), sg.Button("Exit", font = ("Courier New", 30, "bold"), pad=(10,0))]],
                                                    size=(1400, 900),element_justification='c')
        mixer.init()
        mixer.music.load('../assets/audio/Start.mp3')
        mixer.music.play()
        self.model = None

        self.tutorial = False;
        while True:
            event, values = self.window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                mixer.stop()
                mixer.quit()
                sys.exit()
            if event == "Tutorial":
                self.tutorial = True
                mixer.stop()
                mixer.quit()
                break
            if event == 'Start':
                mixer.stop()
                mixer.quit()
                break

        self.window.close()
        self.execute_state_machine()

    def execute_state_machine(self):

        skip_scan = False
        #self.create_init_window()
        found = init_state.find_human()
        try:
            while 1:
                if not found:
                    raise Exception("human not found")
                if not skip_scan:
                    self.query_state_window()
                    state = query_state.scan_human()
                self.create_play_song_window()
                song = select_song_state.select_song(state)[0]
                reward = play_song_state.play_song(song)
                print(state)
                if not skip_scan:
                    history, dqn = training_state.train(state, reward)
                    self.model = dqn
                answer = self.questioning_window()
                # answer = questioning_state.decideForState()

                if answer == 'down':
                    print('Okay another round')
                    skip_scan = False
                elif answer == 'up':
                    print('Okay here comes another song to your mood')
                    skip_scan = True
                elif answer == 'stop':
                    print('Okay Goodbye! I am happy to see u again!')
                    if self.model is not None:
                        print('saving model ...')
                        self.model.save('../results/model/model_{}'.format(NAME))
                        print('model saved')
                        self.window.close()
                        sys.exit()
                self.round += 1
        except SystemExit:
            self.window.close()
            sys.exit()
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            self.window.close()
            self.__init__()

    def create_play_song_window(self):
        self.window = sg.Window("Jukebox", layout=[[sg.Text("I think I know which song fits to your mood :)",font = ("Courier New", 30, "bold"), pad=(0,30))],
                                                   [sg.Image('../assets/smileys/laugh.png', size=(500, 500))],
                                                   [sg.Button("Continue",font = ("Courier New", 30, "bold"), pad=(20,0)), sg.Button("Exit",font = ("Courier New", 30, "bold"), pad=(20,0))],
                                                   ],size=(1400, 900),element_justification='c')

        mixer.init()
        if self.tutorial:
            mixer.music.load('../assets/audio/play song state A.mp3')
            mixer.music.play()

        else:
            mixer.music.load('../assets/audio/play song state B.mp3')
            mixer.music.play()

        while True:
            event, values = self.window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                if self.model is not None:
                    print("model saved")
                    self.model.save('../results/model/model_{}'.format(NAME))
                self.window.close()
                sys.exit()
            if event == 'Continue':
                break
        mixer.stop()
        mixer.quit()
        self.window.close()


    def query_state_window(self):
        self.window = sg.Window("Jukebox", layout=[[sg.Text("I am ready to scan your face and gesture to detect your mood!",font = ("Courier New", 30, "bold"), pad=(0,30))],
                                                   [sg.Image('../assets/smileys/laugh.png', size=(500, 500))],
                                                   [sg.Button("Continue",font = ("Courier New", 30, "bold"), pad=(20,0)), sg.Button("Exit",font = ("Courier New", 30, "bold"), pad=(20,0))],
                                                   ],size=(1400, 900),element_justification='c')
        mixer.init()
        if self.tutorial:
            mixer.music.load('../assets/audio/init state A.mp3')
        else:
            mixer.music.load('../assets/audio/init state B.mp3')

        mixer.music.play()
        while True:
            print("hello")
            event, values = self.window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                if self.model is not None:
                    self.model.save('../results/model/model_{}'.format(NAME))
                sys.exit()
            if event == 'Continue':
                break
        mixer.stop()
        mixer.quit()
        self.window.close()

    def create_init_window(self):
        self.window = sg.Window("Jukebox", layout=[[sg.Text("Iniittttt",font = ("Courier New", 30, "bold"), pad=(0,30))],
                                                   [sg.Image('../assets/smileys/smilelaugh.png', size=(500, 500))],
                                                   [sg.Button("Continue",font = ("Courier New", 30, "bold"), pad=(20,0)), sg.Button("Exit",font = ("Courier New", 30, "bold"), pad=(20,0))],
                                                   ],size=(1400, 900),element_justification='c')

        while True:
            event, values = self.window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                if self.model is not None:
                    self.model.save('../results/model/model_{}'.format(NAME))
                sys.exit()
            if event == 'Continue':
                break

        self.window.close()

    def questioning_window(self):
        self.window = sg.Window("Jukebox", layout=[[sg.Text("I hope you liked the song I played.",font = ("Courier New", 30, "bold"), pad=(0,30))],
                                                   [sg.Image('../assets/smileys/smilelaugh.png', size=(500, 500))],
                                                   [sg.Button("Complete New Round",font = ("Courier New", 30, "bold"), pad=(10,0)), sg.Button("Play new Song",font = ("Courier New", 30, "bold"), pad=(20,0)), sg.Button("Exit",font = ("Courier New", 30, "bold"), pad=(10,0))],
                                                   ],size=(1400, 900),element_justification='c')
        mixer.init()
        if self.tutorial:
            mixer.music.load('../assets/audio/questioning state A.mp3')
        else:
            mixer.music.load('../assets/audio/questioning state B.mp3')

        mixer.music.play()

        while True:
            event, values = self.window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                self.window.close()
                return 'stop'
            if event == "Complete New Round":
                mixer.stop()
                mixer.quit()
                self.window.close()
                self.tutorial = False
                return 'down'
            if event == "Play new Song":
                mixer.stop()
                mixer.quit()
                self.tutorial = False
                self.window.close()
                return 'up'


gui = gui()
