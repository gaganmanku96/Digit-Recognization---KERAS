# import os
# os.environ['KERAS_BACKEND']='plaidml.keras.backend'
from keras.datasets import mnist
from keras.utils import to_categorical
from model import Model
import scipy.ndimage
from keras.preprocessing.image import ImageDataGenerator


# Load data.
(train_x,train_y),(test_x,test_y) = mnist.load_data()

# Normalize the train and test data.
train_x = train_x/255.0
test_x  = test_x/255.0

# Reshape train_x and train_y as (-1,28,28,1)
# -1 will be replaced by batchsize.
# 28,28 is the dimension of the image.
# 1 is depth of the image. We have grayscale images by default.
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)

# one-hot-encode the train labels
train_y = to_categorical(train_y)

# Reshaping test data
test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2],1)

# one-hot-encode the test labels
test_y = to_categorical(test_y)



model = Model()
# We are using categorical_crossentropy since there are multiple classes in the label.
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=128,epochs = 20)

model.save('model.h5')

eval = model.evaluate(test_x,test_y)
print(eval)