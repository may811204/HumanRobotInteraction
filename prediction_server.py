from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

def model_generate(input_shape):
    model = Sequential()
    model.add(Conv2D(64, 5, 5, border_mode="valid", input_shape=input_shape))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2)))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model

model = model_generate((48, 48, 1))
model.load_weights('best_model.6562.hdf5')