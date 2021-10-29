import cv2
from keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from pathlib import Path


class Network:
    def __init__(self, action_shape, state_shape, load=False):

        print(state_shape)
        self.model = Sequential()

        self.model.add(Dense(1, input_dim=17, activation='relu'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))
        self.model.add(Dense(action_shape, activation='softmax'))
        opt = Adam(lr=0.0005)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        if load:
            self.model.load_weights(str(Path(__file__).parent.parent) + '/model/model_weights.h5')

        print(self.model.summary())
    def train(self, state, reward):
        self.model.fit(x=state, y=reward, verbose=1)

    def get_model(self):
        return self.model

    def predict(self, state):
        return self.model.predict(state)