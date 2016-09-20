require 'image'
require 'torch'

local py = require('fb.python')


function imageTest (img)
    print("test1")
    print(type(img))
    img = image.vflip(img)
    print("test2")
    print(type(img))
    return img
end
