# import os
# os.environ['KERAS_BACKEND']='plaidml.keras.backend'
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential

def Model():
    # Sequential Model creates a stack of layers in Linear fashion.
    model = Sequential()

    # In the first Conv2D layer we are using 32 5X5 filters with 28X28 as input shape of image.
    model.add(Conv2D(32,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))

    # Applying maxpooling2d operation to capture only essential features. The filter size is 2 with stride 1.
    model.add(MaxPooling2D())

    # In the second Conv2D layer we are using 64 5X5 filters.
    model.add(Conv2D(64,(5,5),padding='same',activation='relu'))

    # Applying maxpooling2d operation.
    model.add(MaxPooling2D())

    model.add(Flatten())

    # Flatten the conv2d layers and feed it to fully connected NN with 1024 neurons in hidden layer.
    model.add(Dense(1024,activation='relu'))

    # Dropout is a regularization method to reduce overfitting.
    model.add(Dropout(0.4))

    # This is the output layer with 10 outputs.
    model.add(Dense(10,activation='softmax'))

    return model