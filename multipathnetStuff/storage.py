
class Video:
    def __init__(self):
        self.frames = []

    def addFrame(self, frame):
        #if type(frame) is Frame:
            #print('test in Video Class!!!#######')
        self.frames.append(frame)
        #else:
            #print(type(frame),'test in Video Class ERROR !!!!!')

    def getFrames(self):
        return self.frames[:]

class Frame:
    def __init__(self):
        self.imgObjs = []

    def addImageObject(self, imgObj):
        #if type(imgObj) is ImageObject:
        self.imgObjs.append(imgObj)

    def getImageObjects(self):
        return self.imgObjs[:]

class ImageObject:
    def __init__(self, label, prob, mask, bBox):
        self.label = label
        self.prob = prob
        self.mask = mask
        self.bBox = bBox

    def getLabel(self):
        return self.label

    def getProb(self):
        return self.prob

    def getMask(self):
        """Gets the mask, a NxN array containing 1's and 0's"""
        return self.mask

    def getBbox(self):
        return self.bBox
