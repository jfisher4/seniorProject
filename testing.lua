require 'image'
require 'torch'

local py = require('fb.python')

ImageTest = {calls = 0,
    imageTest = function  (self, img)
        self.calls = self.calls + 1
        if self.calls % 2 == 0 then
            img2 = image.flip(img,2)
            print("test1")
        else
            img2 = image.flip(img,1)

            print("test2")
        end


        return img2
    end
    }
