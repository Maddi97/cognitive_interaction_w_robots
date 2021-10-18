import cv2
from keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path

class Network:
    def __init__(self, load=True):
        self.model = Sequential()

        # Initialising the CNN
        # 1 - Convolution
        self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 2nd Convolution layer
        self.model.add(Conv2D(128, (5, 5), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 3rd Convolution layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 4th Convolution layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Flattening
        self.model.add(Flatten())

        # Fully connected layer 1st layer
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        # Fully connected layer 2nd layer
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(7, activation='softmax'))

        opt = Adam(lr=0.0005)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        if load:
            self.model.load_weights(str(Path(__file__).parent.parent) + '/model/model_weights.h5')

        self.train_dir = 'data/train'
        self.val_dir = 'data/test'

        self.num_train = 28709
        self.num_val = 7178
        self.batch_size = 64
        self.num_epoch = 50

    def train(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(48, 48),
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(48, 48),
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode='categorical')

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
        model_info = self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.num_train // self.batch_size,
            epochs=self.num_epoch,
            validation_data=validation_generator,
            validation_steps=self.num_val // self.batch_size)
        # self.plot_model_history(model_info)
        self.model.save_weights('model1.h5')

    def get_model(self):
        return self.model
