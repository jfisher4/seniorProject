import cv2
import numpy as np
import lutorpy as lua
require("testing")

lua.LuaRuntime(zero_based_index=True)



image2 = cv2.imread('/home/ryanubuntu/Documents/seniorProject/res.jpg',1)
print(type(image2))
cv2.imshow("initial", image2)

x = torch.fromNumpyArray(image2)

w = imageTest(x)
#print(w)

y = w.asNumpyArray()
cv2.imshow("result", y)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',image)
    cv2.destroyAllWindows()
