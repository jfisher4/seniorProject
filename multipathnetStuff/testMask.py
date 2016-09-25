import cv2
import numpy as np
import pickle

#pickle.dump( masks, open( "/home/robotics_group/multipathnet/masksTest.p", "wb" ) )
masks = pickle.load( open( "/home/ryanubuntu/multipathnet/masksTest.p", "rb" ) )

print(masks.shape, "masks.shape")

newList = []

counter = 0

for i in range(masks.shape[1]):
    for j in range(masks.shape[2]):
        if masks[0][i][j] != 0:
            #print(masks[0][i][j])
            #newList.append(masks[0][i][j])
            newList.append(255) # convert the one to 255 for presentation in image
            counter = counter + 1
        else:
            #newList.append(masks[0][i][j])
            newList.append(0) # convert anything else to 0
        #newMasks[i][j] = masks[i][j]

print (counter, "counter")

newMasks = np.array(newList,dtype = np.uint8).reshape(masks.shape[1],masks.shape[2]) # convert to uint8 array
print(newMasks.shape)
print(newMasks)

threshed = cv2.adaptiveThreshold(newMasks, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0) # this preserves the outer boundary of the masked blob
cv2.imshow('newMasks',newMasks)
cv2.imshow('threshed',threshed)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

