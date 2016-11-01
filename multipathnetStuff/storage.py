class Video(object):
    def __init__(self):
        self._frames = []
        self._size = [0,0]

    def addFrame(self, frame):
        if type(frame) is Frame:
            self._frames.append(frame)

    def getFrames(self):
        return self._frames[:]

    def getSize(self):
        return (self._size[0],self._size[1])

class Frame(object):
    def __init__(self):
        self._imgObjs = []

    def addImageObject(self, imgObj):
        if type(imgObj) is ImageObject:
            self._imgObjs.append(imgObj)

    def getImageObjects(self):
        return self._imgObjs[:]

class ImageObject(object):
    def __init__(self, label, prob, mask, bBox):
        self._label = label
        self._prob = prob
        self._mask = mask
        self._bBox = bBox

    def getLabel(self):
        return self._label

    def getProb(self):
        return self._prob

    def getMask(self):
        """Gets the mask, a NxN array containing 1's and 0's"""
        return self._mask

    def getBbox(self):
        return self._bBox
