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
import data_pre_processor

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

img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 1200
img_channels = 1

Train_x, Train_y, Val_x, Val_y = data_pre_processor.load_data()

Train_x = np.asarray(Train_x) 
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = np.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

from keras import backend as K
if K._image_dim_ordering == 'th':
    Train_x = Train_x.reshape(Train_x.shape[0], 1, img_rows, img_cols)
    Val_x = Val_x.reshape(Val_x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols, 1)
    Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = model_generate(input_shape)

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    samples_per_epoch=Train_x.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[checkpointer])

