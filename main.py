import cv2
import numpy as np 
import time
from model import Model

model = Model()
model.load_weights('new_model.h5')
drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

print('Instructions:\n\t1. Press P to Predict Number\n\t2. Press R to Reset screen\n\t3. Press Q to quit')

# mouse callback function
def draw(event,x,y,flags,param):
    global current_x,current_y,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_x,current_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(im,(current_x,current_y),(x,y),(255,255,255),5)
            current_x = x
            current_y = y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(im,(current_x,current_y),(x,y),(255,255,255),5)
        current_x = x
        current_y = y
    return x,y    



im = np.zeros((128,128,1))
cv2.namedWindow("InputWindow")
cv2.setMouseCallback('InputWindow',draw)
while(1):
    # time.sleep(0.08)
    im=im.astype(np.uint8)
    cv2.imshow('InputWindow',im)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('r'):
        im = np.zeros((128,128,1))
    elif k == ord('p'):
        img = cv2.resize(im,(28,28))
        img = img.reshape(1,28,28,1)
        pred = model.predict(img)
        max = 0
        index=0
        # cv2.imshow('Output',img.reshape(28,28))
        print(*pred)
        for i,val in enumerate(list(*pred)):
            if(val>max):
                max,index = val,i
        print(index)

cv2.destroyAllWindows()