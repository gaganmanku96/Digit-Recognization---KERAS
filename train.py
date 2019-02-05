# import os
# os.environ['KERAS_BACKEND']='plaidml.keras.backend'
from keras.datasets import mnist
from keras.utils import to_categorical
from model import Model
import scipy.ndimage
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=30)

(train_x,train_y),(test_x,test_y) = mnist.load_data()

train_x = train_x/255.0
test_x  = test_x/255.0

train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
train_y = to_categorical(train_y)

test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2],1)
test_y = to_categorical(test_y)



model = Model()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(train_x,train_y,batch_size=128),steps_per_epoch = 128,epochs = 20)

model.save('model.h5')

eval = model.evaluate(test_x,test_y)
print(eval)