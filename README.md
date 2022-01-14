# Project Cognitive Interaction with Robots


### Install required packages

```
pip install -r requirements.txt
```

### Run JukeBot with graphical user interface

```
python gui.py
```

### Set parameters for the run

In gui/gui.py in lines 21-24 you can set the parameters for the run
parameter:
1. NAME: choose a name of a user -> model for this user is saved and can later be loaded
2. LOAD: set True if there is already a pre-trained model corresponding to the NAME parameter and you want to continue with this model. To train a completely new model set it to False
3. FIELD_EXPERIMENT: set True if the run is a field experiment or you want to really use the juke bot: False is only for the simulation run
4. RANDOM: if it is the control group set it to True, then only random songs are selected. False uses the predictions of the neural network


###Results und Models

In the directory 'results' all results can be found:
1. /models_participants' contains all trained model of the participants in the field experiment and can be loaded and retrained
2. /simulation contains the model of the simulation run
3. plots.py is the script used for visualize the simulation results
4. the *.csv files contains the collected data of the simulation runs

### Code explanation

1. In the folder 'control_unit' are the implementation of the state maschine and all according states
2. folder 'rl' contains all files of the implementation of the reinforcement learning algorithm
3. in folder 'tracking' are all files related to the detection of the facial expressin and the gesture
