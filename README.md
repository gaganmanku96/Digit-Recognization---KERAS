# Digit-Recognization---KERAS

### This project uses MNIST dataset and opencv to predict the digit that is drawn in the input window.

### Requirements
```
1. opencv
2. Keras
3. Tensorflow
4. Numpy
```

### Understanding code
```
drawing = False
def draw(event,x,y,flags,param):
    global current_x,current_y,drawing

    # If the left mouse key is continiously pressed then drawing will be set to True.
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_x,current_y=x,y
    
    # If drawing == True and mouse is moved then line will be drawn.
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(im,(current_x,current_y),(x,y),(255,255,255),5)
            current_x = x
            current_y = y
            
    # If left mouse key is released then drawing will be set to False.        
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(im,(current_x,current_y),(x,y),(255,255,255),5)
        current_x = x
        current_y = y
    return x,y    
```


```
im = np.zeros((128,128,1))
# Creating a black image to draw numbers on it.

cv2.namedWindow("InputWindow")
# create a window.

cv2.setMouseCallback('InputWindow',draw)
# Sets mouse handler for the specified window i.e. 'InputWindow'

while(1):
    im=im.astype(np.uint8)
    # The default dtype for np.zeros is float. We need to change it to uint8.
    
    cv2.imshow('InputWindow',im)
    # Displays the image on the specified window.

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.destroyAllWindows()
```


Creating CNN Model
```
def Model():
    model = Sequential()
    # Sequential Model creates a stack of layers in Linear fashion.
    
    model.add(Conv2D(32,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
    # In the first Conv2D layer we are using 32 5X5 filters with 28X28 as input shape of image.
    
    model.add(MaxPooling2D())
    # Applying maxpooling2d operation to capture only essential features. The filter size is 2 with stride 1.

    model.add(Conv2D(64,(5,5),padding='same',activation='relu'))
    # In the second Conv2D layer we are using 64 5X5 filters.
    
    model.add(MaxPooling2D())
    # Applying maxpooling2d operation.

    model.add(Flatten())

    model.add(Dense(1024,activation='relu'))
    # Flatten the conv2d layers and feed it to fully connected NN with 1024 neurons in hidden layer.
    
    model.add(Dropout(0.4))
    # Dropout is a regularization method to reduce overfitting.

    model.add(Dense(10,activation='softmax'))
    # This is the output layer with 10 outputs.

    return model
```


Training the Network
```
from keras.datasets import mnist
from keras.utils import to_categorical
from model import Model

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
model.fit(train_x,train_y,batch_size=128 ,epochs = 20)

model.save('model.h5')

eval = model.evaluate(test_x,test_y)
print(eval)
```

Modifying the initial code and load the saved model.
