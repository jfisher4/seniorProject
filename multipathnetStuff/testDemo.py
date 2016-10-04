import cv2
import numpy as np
import lutorpy as lua
import pickle
#import the lua script
require("demo")
ROI_RESIZE_DIM = (600,400)
lua.LuaRuntime(zero_based_index=False)

def test(img):
    # read in an image
    image2 = cv2.imread('/home/robotics_group/seniorProject/res.jpg',1)
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


image2 = cv2.imread('/home/robotics_group/seniorProject/res.jpg',-1) #0 for greyscale, 1 for color, -1 for unchanged

saveName="/home/robotics_group/multipathnet/deepmask/data/testTmp.jpeg" # save the file so that we can load it into lua, hopefully we can find a better way
cv2.imwrite(saveName, image2)

print(image2.shape)
image2 = np.reshape(image2, (image2.shape[2],image2.shape[0],image2.shape[1]))#convert the shape to  the same as lua load image, but there is still somethign wrong with the data
image2 = cv2.normalize(image2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
luaImg = torch.fromNumpyArray(image2)
print(image2.shape)
print(luaImg.size(luaImg))
# create an instance of the MultiPathNet Object
multiPathObject = MultiPathNet
#init the object
multiPathObject.init(multiPathObject)
multiPathObject.start(multiPathObject)
probs, names, masks = multiPathObject.processImg(multiPathObject,luaImg)
# convert the torch tensor back to a numpy array
masks = masks.asNumpyArray()
#pickle.dump( masks, open( "/home/robotics_group/multipathnet/masksTest.p", "wb" ) ) #used for offline testing to save time
print(masks.shape, "masks.shape")
newList = [] # new list for getting data from masks since my attempts to assign vals in a numpy array were not working
#counter = 0
#for i in range(masks.shape[1]):
    #for j in range(masks.shape[2]):
        #if masks[0][i][j] != 0:
            ##print(masks[0][i][j])
            ##newList.append(masks[0][i][j])
            #newList.append(255) # convert the one to 255 for presentation in image
            #counter = counter + 1
        #else:
            ##newList.append(masks[0][i][j])
            #newList.append(0) # convert anything else to 0
        ##newMasks[i][j] = masks[i][j]
masks = cv2.normalize(masks, None, 0, 255, cv2.NORM_MINMAX)
#print (counter, "counter")

newMasks = np.array(newList,dtype = np.uint8).reshape(masks.shape[1],masks.shape[2]) # convert to uint8 array of the same dim as the image
print(newMasks.shape, "newMasks.shape")

cv2.imshow('newMasks',newMasks)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',image)
    cv2.destroyAllWindows()
