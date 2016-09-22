import cv2
import numpy as np
import lutorpy as lua
#import pickle
#import the lua script
require("demo")

lua.LuaRuntime(zero_based_index=True)

def test(img):
    # read in an image
    image2 = cv2.imread('/home/ryanubuntu/seniorProject/res.jpg',1)
    # print the image type
    #print(type(image2))
    # show the original image
    #cv2.imshow("initial", image2)

    #convert the image to a torch tensor
    x = img.asNumpyArray()
    cv2.imshow('window', x)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',image)
        cv2.destroyAllWindows()


image2 = cv2.imread('/home/ryanubuntu/seniorProject/res.jpg',1)
x = torch.fromNumpyArray(image2)
# create an instance of the MultiPathNet Object
w = MultiPathNet
#print(type(w))
w.init(w)
w.start(w)
probs, names, masks = w.processImg(w,x)
#masks = masks.asNumpyArray()
#cv2.imshow(tmp, masks)


# convert the torch tensor back to a numpy array
#y = x.asNumpyArray()
#cv2.imshow("result", y)
# wait for user input to exit and close windows

