# import os
# os.environ['KERAS_BACKEND']='plaidml.keras.backend'
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential

def Model():
    model = Sequential()
    model.add(Conv2D(32,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(64,(5,5),padding='same',activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(10,activation='softmax'))

    return model