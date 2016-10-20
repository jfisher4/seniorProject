
class Video(object):
    def __init__(self):
        self._frames = []

    def addFrame(self, frame):
        if type(frame) is Frame:
            self._frames.append(frame)

    def getFrames(self):
        return self._frames[:]

class Frame(object):
    def __init__(self):
        self._imgObjs = []

    def addImageObject(self, imgObj):
        if type(imgObj) is ImageObject:
            self._imgObjs.append(imgObj)

    def getImageObjects(self):
        return self._imgObjs[:]

class ImageObject(object):
    def __init__(self, label, prob, mask):
        self.label = label
        self.prob = prob
        self.mask = mask

    def getLabel(self):
        return self.label

    def getProb(self):
        return self.prob

    def getMask(self):
        """Gets the mask, a NxN array containing 1's and 0's"""
        return self.mask
