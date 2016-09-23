--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- DeepMask + MultiPathNet demo

require 'deepmask.SharpMask'
require 'deepmask.SpatialSymmetricPadding'
require 'deepmask.InferSharpMask'
require 'inn'
require 'fbcoco'
require 'image'
model_utils = require 'models.model_utils'
utils = require 'utils'
coco = require 'coco'

MultiPathNet = {np=5,si=-2.5,sf=.5,ss=.5,dm=false,thr=0.5,maxsize=600,sharpmask_path='/home/robotics_group/multipathnet/data/models/sharpmask.t7',
    multipath_path='/home/robotics_group/multipathnet/data/models/resnet18_integral_coco.t7'}
--print(MultiPathNet.sharpmask_path)
MultiPathNet.init = function (self)
    print("starting multipathnet init")
    self.sharpmask = torch.load(self.sharpmask_path).model
    self.multipathnet = torch.load(self.multipath_path)
    self.meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}
    self.scales = {}
    for i = self.si,self.sf,self.ss do table.insert(self.scales,2^i) end

    self.infer = Infer{
        np = self.np,
        scales = self.scales,
        meanstd = self.meanstd,
        model = self.sharpmask,
        dm = self.dm,
    }
    print("multipathnet init finished")
end
MultiPathNet.start = function (self)
    print("starting multipathnet start")
    self.sharpmask:inference(self.np)
    self.multipathnet:evaluate()
    self.multipathnet:cuda()
    model_utils.testModel(self.multipathnet)
    self.detector = fbcoco.ImageDetect(self.multipathnet, model_utils.ImagenetTransformer())

    print("printing scales in start method", self.scales)
    --self.infer.scales = self.scales -- update the scales
    print("multipathnet start finished")
end
MultiPathNet.processImg = function (self, img)
    print("multipathnet processImg starting")
    print(img:type())
    print(img:size())
    self.test(self,img)
    img2 = img:clone()
    --local img2 = image.load('/home/robotics_group/multipathnet/deepmask/data/testImage.jpg')
    --local img2 = self.clampImage(self, img2)
    img2 = image.scale(img2, self.maxsize)
    --img2 = img:float()
    print(type(img2))
    h,w = img2:size(2),img2:size(3)
    print("test1")
    --print("self.infer:forward",self.infer:forward())
    self.infer:forward(img2)  -- this is where the problem is
    print("test2")
    masks,_ = self.infer:getTopProps(.2,h,w)

    Rs = coco.MaskApi.encode(masks)
    bboxes = coco.MaskApi.toBbox(Rs)
    print("test3")
    bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
    detections = self.detector:detect(img2:float(), bboxes:float())
    prob, maxes = detections:max(2)
    -- remove background detections
    print("test4")
    idx = maxes:squeeze():gt(1):cmul(prob:gt(self.thr)):nonzero():select(2,1)
    bboxes = bboxes:index(1, idx)
    maxes = maxes:index(1, idx)
    prob = prob:index(1, idx)
    print("test5")
    scored_boxes = torch.cat(bboxes:float(), prob:float(), 2)
    final_idx = utils.nms_dense(scored_boxes, 0.3)
    -- remove suppressed masks
    masks = masks:index(1, idx):index(1, final_idx)
    print("test6")
    dataset = paths.dofile'./DataSetJSON.lua':create'coco_val2014'
    print("test7")
    --make a table to hold all of the names
    names = {}
    --coco.MaskApi.drawMasks(res, masks, 10)
    for i,v in ipairs(final_idx:totable()) do
        class = maxes[v][1]-1
        name = dataset.dataset.categories[class]
        -- insert the name into the table
        table.insert(names,name)
        print(prob[v][1], class, name)

    end

    print('| done')
    return prob, names, masks
end

MultiPathNet.clampImage =  function (self, tensor)
   --if tensor:type() == 'torch.ByteTensor' then
      --return tensor
   --end
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   a.image.saturate(a) -- bound btwn 0 and 1
   a:mul(255)          -- remap to [0..255]
   return a
end
MultiPathNet.test =  function (self, img)
    print(img:type(), 'testing!!!!')
end
