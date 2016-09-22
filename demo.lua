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
local model_utils = require 'models.model_utils'
local utils = require 'utils'
local coco = require 'coco'

MultiPathNet = {np=5,si=-2.5,sf=.5,ss=.5,dm=false,thr=0.5,maxsize=600,sharpmask_path='/home/ryanubuntu/multipathnet/data/models/sharpmask.t7',
    multipath_path='/home/ryanubuntu/multipathnet/data/models/resnet18_integral_coco.t7'}
--print(MultiPathNet.sharpmask_path)
MultiPathNet.sharpmask = torch.load(MultiPathNet.sharpmask_path).model
MultiPathNet.multipathnet = torch.load(MultiPathNet.multipath_path)

MultiPathNet.meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}
MultiPathNet.scales = {}
MultiPathNet.infer = Infer{
    np = MultiPathNet.np,
    scales = MultiPathNet.scales,
    meanstd = MultiPathNet.meanstd,
    model = MultiPathNet.sharpmask,
    dm = MultiPathNet.dm,
}
MultiPathNet.start = function (self)
    self.sharpmask:inference(self.np)
    self.multipathnet:evaluate()
    self.multipathnet:cuda()
    model_utils.testModel(self.multipathnet)
    self.detector = fbcoco.ImageDetect(self.multipathnet, model_utils.ImagenetTransformer())
    for i = self.si,self.sf,self.ss do table.insert(self.scales,2^i) end
    print(scales)
    self.infer.scales = self.scales -- update the scales
end
MultiPathNet.processImg = function (self, img)
    img = image.scale(img, self.maxsize)
    h,w = img:size(2),img:size(3)
    self.infer:forward(img)
    masks,_ = self.infer:getTopProps(.2,h,w)
    Rs = coco.MaskApi.encode(masks)
    bboxes = coco.MaskApi.toBbox(Rs)
    bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
    detections = self.detector:detect(img:float(), bboxes:float())
    prob, maxes = detections:max(2)
    -- remove background detections
    idx = maxes:squeeze():gt(1):cmul(prob:gt(self.thr)):nonzero():select(2,1)
    bboxes = bboxes:index(1, idx)
    maxes = maxes:index(1, idx)
    prob = prob:index(1, idx)
    scored_boxes = torch.cat(bboxes:float(), prob:float(), 2)
    final_idx = utils.nms_dense(scored_boxes, 0.3)
    -- remove suppressed masks
    masks = masks:index(1, idx):index(1, final_idx)
    dataset = paths.dofile'./DataSetJSON.lua':create'coco_val2014'
    --make a table to hold all of the names
    names = {}
    coco.MaskApi.drawMasks(res, masks, 10)
    for i,v in ipairs(final_idx:totable()) do
        class = maxes[v][1]-1
        name = dataset.dataset.categories[class]
        -- insert the name into the table
        table.insert(names,name,i)
        print(prob[v][1], class, name)

    end

    print('| done')
    return prob, names, masks
end

