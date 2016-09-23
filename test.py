import cv2
import numpy as np
import lutorpy as lua
#import the lua script
require("testing")

lua.LuaRuntime(zero_based_index=True)

# read in an image
image2 = cv2.imread('/home/robotics_group/seniorProject/res.jpg',1)
# print the image type
print(type(image2))
# show the original image
cv2.imshow("initial", image2)

#convert the image to a torch tensor
x = torch.fromNumpyArray(image2)

# create an instance of the imageTest Object
w = ImageTest
print(type(w))

# create a loop to test the lua object
for i in range(5):
    tmp = "window_" + str(i)
    x = w.imageTest(w, x)
    y = x.asNumpyArray()
    cv2.imshow(tmp, y)
# print the lua object attribute calls which tracks how many times we call the image test function
print(w.calls)

# convert the torch tensor back to a numpy array
#y = x.asNumpyArray()
cv2.imshow("result", y)
# wait for user input to exit and close windows
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',image)
    cv2.destroyAllWindows()
