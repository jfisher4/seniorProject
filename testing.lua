-- test script for passing data back and forth from lua
require 'image'
local py = require('fb.python')


function imageTest (img)
    --local img = image.load(img)
    --image.display{img}
    print("test")

    return img
end
