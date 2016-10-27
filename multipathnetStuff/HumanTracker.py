import cv2
import numpy as np
#import lutorpy as lua
import pickle
import gzip
#import time
from storage import *
#import os


class HumanTracker:

    def __init__(self, directory, videoname):
        self.directory = directory
        self.videoname = videoname
        self.cap = cv2.VideoCapture(directory+videoname)
        self.metadata = videoname.split("_")
        self.frameNumber = 0
        f = gzip.open( directory+videoname+".pklz", "rb" )
        self.videoObj = pickle.load(f)
        f.close()
        print(len(self.videoObj.getFrames()), 'length of get frames of restored video object')
        self.trackedPeople = People()
        self.ROI_RESIZE_DIM = (600,337)
            
    def readAndTrack(self):
        #time1 = time.time()
        ret,img = self.cap.read()
        
        if not ret: #allow for a graceful exit when the video ends
            print("Exiting Program End of Video...")
            self.cap.release()
            cv2.destroyAllWindows()
            return(None, 0) #return 0 to toggle active off
        img = cv2.resize(img,self.ROI_RESIZE_DIM)
        imgDisplay = img.copy()
        self.videoObjCurrentObjs = self.videoObj.getFrames()[self.frameNumber].getImageObjects()
        if self.videoObjCurrentObjs[0].getMask() != None: # for some reason there exists some none objects in the frames image object list. 
            detPersonList = []
            for i in range(len(self.videoObjCurrentObjs)):
                if self.videoObjCurrentObjs[i].getLabel() != None:
                    #print(self.videoObjCurrentObjs[i].getLabel())
                    currentMask = cv2.normalize(self.videoObjCurrentObjs[i].getMask(), None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imshow("mask_"+str(i),currentMask)
                    #people class stuff
                    if self.videoObjCurrentObjs[i].getLabel() == 'person':
                        hist = getHist(img,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM)
                        bBox = self.videoObjCurrentObjs[i].getBbox()
                        bBox = bBox.astype(int)
                        bBox = [bBox[0],bBox[1],bBox[2]-bBox[0],bBox[3]-bBox[1]] #convert from x1,y1,x2,y2 to x,y,w,h
                        tmpPerson = Detection(self.videoObjCurrentObjs[i].getMask(), bBox, self.videoObjCurrentObjs[i].getLabel(), self.videoObjCurrentObjs[i].getProb(), hist)
                        detPersonList.append(tmpPerson)
                        displayHistogram(hist,self.frameNumber,i)
                        
                        cv2.rectangle(imgDisplay, (bBox[0], bBox[1]), (bBox[0]+bBox[2],bBox[1]+bBox[3]), (255,0,0), 4)
                        
                        #bBox = newBox
                        #print(bBox,"bBox")
                        
                                        
                    else:
                        print("Not a person")                                                      
                    
                else:
                    print(self.videoObjCurrentObjs[i].getLabel()," the label of problem object")
                    #print(type(self.videoObjCurrentObjs[i].getMask()))
            self.trackedPeople.update(img,self.frameNumber,detPersonList) 
        else:
            print("Empty frame objects this frame")
        
        self.trackedPeople.refresh(img,imgDisplay,self.frameNumber,self.ROI_RESIZE_DIM) #update all of the people
        for person in self.trackedPeople.listOfPeople:
            if person.V == 1:# HOG has updated visibility this frame
                
                cv2.rectangle(imgDisplay, (person.fX, person.fY), (person.fX+person.fW,person.fY+person.fH), (0,0,255), 2)
                cv2.putText(imgDisplay,str(person.ID),(person.fX+5,person.fY+30),0, 1, (0,0,255), 3,8, False)
            elif person.V < 1000: #HOG did not update visibility and person was tracked with background subtracction
                cv2.rectangle(imgDisplay, (person.fX, person.fY), (person.fX+person.fW, person.fY+person.fH), (0,255,0), 2) #show meanshift roi box green
                cv2.putText(imgDisplay,str(person.ID),(person.fX+5,person.fY+30),0, 1, (0,255,0), 3,8, False)
        

            
        height, width = img.shape[:2]
        printString = 'Frame ' + str(self.frameNumber)
        cv2.putText(imgDisplay,printString,(20,80),0,1, (0,0,255),3,8,False)
        cv2.imshow(self.metadata[0],imgDisplay) 
        #cv2.imshow("hsv",cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        
        print('framenumber ' + str(self.frameNumber))
        self.frameNumber += 1     
        k = cv2.waitKey(2) & 0xFF
        if k == ord('p'):
            print("Pausing...")
            return (None,2) #return 2 for paused
        elif k == ord('q'):
            print("Exiting Program...")
            
            self.cap.release()
            cv2.destroyAllWindows()
            return (None,0) #return 0 to toggle active off
        #elif self.frameNumber == 10000: #for testing only to pause at a certain frame
            #timeEnd = time.time()
            #totalTime = timeEnd - timeStart
            #print(totalTime,'totalTime')
        elif self.frameNumber == 10: #for testing only to pause at a certain frame
            return (None,2)
        return (None,1) #return 1 to stay active

class People():
    ## The constructor.
    def __init__(self):
        self.listOfPeople=list()
        self.lostListOfPeople=list()
        self.index=0
        #self.trackedPeople.update(img,self.fgmask,fX,fY,fW,fH,self.frameNumber,roi_hist,self.trackedPeople.listOfPeople)
# Updates an item in the list of people/object or appends a new entry or assigns to a group or removes from a group
    def update(self,img,frameNumber,detectionsList):
        personList = self.listOfPeople
        
        if len(personList) == 0:
            i = 0
            for detection in detectionsList:
                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist()) #step 3 only update persons histogram on creation, not in subsequent updates.
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                print("creating person for the first time!!",i)
                i += 1
        
        else:
            i = 0
            for person in personList:
                bestMatch = []
                j = 0
                for detection in detectionsList:
                    
                    box0 = detection.getBbox()
                    box1 = [box0[0],box0[1],box0[0]+box0[2],box0[1]+box0[3]]
                    box2 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
                    lapping = overLap(box1,box2) 
                    p1 = [box0[0]+box0[2]/2,box0[1]+box0[3]/2]
                    p2 = [person.fX + person.fW/2,person.fY+person.fH/2]
                    pixelDist = objectDistance(p1,p2)
                    print(pixelDist, " Pixel Dist")
                    #print (lapping, "lapping")
                    histDist = histogramComparison(detection.getHist(),person.hist)
                    #print(histDist, "histDist")
                    if len(bestMatch)>0:
                        if lapping > 0.0 or pixelDist < 150:
                            if histDist < bestMatch[3] :
                                bestMatch = [lapping, i, j, histDist]  
                    else: #used in first iteration to set up overlap and histogram comparison
                        if lapping > 0.0 or pixelDist < 150:
                            bestMatch = [lapping, i, j, histDist]
                    j = j + 1
            
                if len(bestMatch)>0:
                    person.V=0
                    person.updateLocation(detectionsList[bestMatch[2]].getBbox())
                    person.hist = detectionsList[bestMatch[2]].getHist()
                    print("updating person ", person.ID)
                    del detectionsList[bestMatch[2]]
                else:
                    print("did not find detection for person ", i)
                i = i + 1
           
            for detection in detectionsList:
                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist()) 
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                print("creating NEW person !!",self.index)
   

    def refresh(self,img,imgCopy,frameNumber,RoiResizeDim): #updates people's boxes and checks for occlusion
        personList = list(self.listOfPeople) #make copy of people list to use for while loop

        while len(personList) > 0:

            person1 = self.getPerson(personList[0].ID,self.listOfPeople)
            #print(person1.ID,'moving = ', person1.moving)
            #flag = 0
            person1.V=person1.V+1
            
            #print(person1.ID, flag, 'flag for current person')  

#            if person1.nearEdge == True and person1.edgeCounter > 15:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
#                
#                self.insertPerson(person1,self.lostListOfPeople)
#                print(person1.ID,'sent to lost people left edge of scene')
#                self.removePerson(person1.ID,self.listOfPeople)
#                personList.remove(person1)
#                continue #skip to next person
#
#            if  person1.roiCounter > 500 and person1.V > 120:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
#                
#                self.insertPerson(person1,self.lostListOfPeople)
#                print(person1.ID,'sent to lost people lost in scene')
#                self.removePerson(person1.ID,self.listOfPeople)
#                personList.remove(person1)
#                continue #skip to next person
#            
#            elif person1.V > 0 and flag == 1:# : # do for every person with no BS roi and shares previous roicurrent
#                person1.roiCounter += 1
#                #print('case2a in refresh, no current ROI, adjust box and meanshift')            
#
#            else: # person.roicurrent != [] and not shared
#                #print('case3 in refresh, person has current ROI, personbox = person.roi',person1.ID)
#                
#                person1.roiCounter = 0
#                

            person1.location.append([frameNumber,(person1.fX+(person1.fX+person1.fW))/2,(person1.fY+(person1.fY+person1.fH))/2])
            

#            if person1.fX <= 25 or person1.fX+person1.fW >= RoiResizeDim[0]-1 or person1.fY+ person1.fH <= 10 or person1.fY+person1.fH >= RoiResizeDim[1]-1 : #check if person1 is on edge of scene
#                person1.nearEdge = True
#                person1.edgeCounter +=1
#            else:
#                person1.nearEdge = False

            personList.remove(person1)


    def insertPerson(self,person,personList): # perhaps a better way to do this or it is unnessessary
        #print(len(personList),'person list length before')
        personList.append(person)
        personList.sort(key=lambda x: x.ID, reverse=False)


    def removePerson(self,personID,personList):
        i = 0
        if len(personList) > 0: #remove correct person from person list
            while i < len(personList):
                currentID = personList[-(i+1)].ID
                if personID == currentID:
                    personList.remove(personList[-(i+1)])
                    break
                i += 1

    def getPerson(self,personID,personList):
        i = 0
        if len(personList) > 0: #remove correct person from person list
            while i < len(personList):
                currentID = personList[-(i+1)].ID
                if personID == currentID:
                    return personList[-(i+1)]
                i += 1
            return []
        else:
            return []

# This class stores all information about a single person/object in the frame.
class Person():

    def __init__(self,ID,bBox,visible,hist):
        self.ID=ID
        self.fX=bBox[0]
        self.fY=bBox[1]
        self.fW=bBox[2]
        self.fH=bBox[3]
        self.V=visible
        self.location=[]
        self.kalmanLocation = []
        self.direction = []
        self.hist = hist
        self.histList = [hist]
        self.kalmanX = KalmanFilter(self.fX+(self.fW/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0)
        self.kalmanY = KalmanFilter(self.fY+(self.fH/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0)
        self.bBox = bBox
        self.lastROICurrent = []
        self.lastGoodROI = []
        self.locationArray = np.array([],ndmin = 2)
        self.locationArray.shape = (0,2)
        self.intersected = False                            #
        self.moving = True                                  #
        self.running = False                                #
        self.speed = -1                                  #
        self.clusterID = 0
        self.leftObject = []                                #            #list of tuples with frame num and location of object
        self.nearEdge = False                               #for detecting that the person is leaving the scene
        self.edgeCounter = 0                                #for detecting that the person is leaving the scene
        self.roiCounter = 0
        self.heading = -1
        self.worldLocation = []
        self.clusterGroup = None
        self.sharedROI = False
    
    def updateLocation(self, bBox):
        self.fX=bBox[0]
        self.fY=bBox[1]
        self.fW=bBox[2]
        self.fH=bBox[3]

class ClusterGroup(): #class to hold the individual groups of occluded people. consider renaming these classes

    def __init__(self,label):
        self.index=0
        self.people=[]
        self.previousLen = 0
        self.label = label
        self.state = 1 #used to get rid of cluster that is empty


    def add(self, person):
        if len(person) <= 1:
            self.people.append(person)
            self.index= self.index + 1
        else:
            self.people.extend(person)
            self.index = self.index + len(person)

    def remove(self, person):
        if len(person) <= 1:
            self.people.remove(person)
            self.index= self.index - 1
        else:
            for person2 in self.people:
                if person2 in person:
                    self.people.remove(person)
                    self.index= self.index - 1



def displayHistogram(histogram,frameNumber=-1,id=-1):
    histogram = histogram.reshape(-1)
    binCount = histogram.shape[0]
    BIN_WIDTH = 3
    img = np.zeros((256, binCount*BIN_WIDTH, 3), np.uint8)
    for i in xrange(binCount):
        h = int(histogram[i])
        cv2.rectangle(img, (i*BIN_WIDTH+1, 255), ((i+1)*BIN_WIDTH-1, 255-h), (int(180.0*i/binCount), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if(frameNumber != -1):
        cv2.putText(img,'Frame#: %d' %frameNumber,(20,20),0, .75, (255,255,255), 1,8, False)
    if(id!=-1):
        cv2.imshow("Person "+str(id)+" Histogram", img)
    else:
        cv2.imshow("Probable Person Histogram", img)

def getHist(img,mask,fX,fY,fW,fH,ROI_RESIZE_DIM): #not a foreground hist
 
    #hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_roi = img
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        
    return roi_hist

def histogramComparison(curHist,newHist):
    distance = cv2.compareHist(curHist,newHist,4) #update based on color match 4
    return distance

def overLap(a,b):  # returns 0 if rectangles don't intersect #a and b = [xmin ymin xmax ymax]  
    areaA = float((a[2]-a[0]) * (a[3]-a[1]))
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3],b[3]) - max(a[1],b[1])
    #print(dx,'dx')
    #print(dy,'dy')
    if (dx > 0 ) and (dy > 0):
        intersect = float(dx*dy)
        if areaA != 0:
            ratioA = intersect/areaA
        else:
            ratioA = 0
        return ratioA

    else:
        return 0 
        
def objectDistance(objectBottom,cameraPosition):
    bottom = np.array((objectBottom[0] ,objectBottom[1], 0))#or use z = 1 if trouble add one to camera height 
    cameraBase = np.array((cameraPosition[0],cameraPosition[1], 0))#or use z = 1
    cameraBaseToBottom = np.linalg.norm(cameraBase-bottom)#1 find distance from base of camera to bottom point
    return float(cameraBaseToBottom)
    
class KalmanFilter():
#Credit to Nicholas Kennedy. Renaming function for simplicity's sake.

    def __init__(self,prediction,kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0):
        self.kalmanGain = 0
        self.covariance = 1.0 #1.0
        self.measurmentNoiseModel = 1.5#.8
        self.prediction = prediction
        self.covarianceGain = 1.10#1.05
        self.lastSensorValue = 0
        
    def updatePrediction(self, sensorValue = None):
        if sensorValue != None:
            self.prediction = self.prediction+self.kalmanGain*(sensorValue-self.prediction)
        else:
            self.prediction = self.prediction+self.kalmanGain*(0-self.prediction)
        #return self.prediction
    
    def updateCovariance(self):
        self.covariance = (1-self.kalmanGain)*self.covariance#(1-self.kalmanGain)*self.covariance
        #print(self.covariance,'covariance')
    
    def updateKalmanGain(self):
        self.kalmanGain = (self.covariance)/(self.covariance+self.measurmentNoiseModel)#(1+self.covariance)/(self.covariance+self.measurmentNoiseModel)
        
    def step(self, sensorValue = None):
        self.covariance = self.covariance * self.covarianceGain
        self.updateKalmanGain()
        self.updatePrediction(sensorValue)
        
        if sensorValue != None:
            self.updateCovariance()
            self.lastSensorValue = sensorValue
        else:
            self.covariance = (1-self.kalmanGain)*self.covariance#(self.kalmanGain)*self.covariance#(1-self.kalmanGain)*self.covariance
            
class Detection():
    def __init__(self, mask, bBox, label, prob, hist):
        self._mask = mask
        self._bBox = bBox
        self._label = label
        self._prob = prob
        self._hist = hist
        
    def getMask(self):
        return self._mask
    
    def getBbox(self):
        return self._bBox
    
    def getLabel(self):
        return self._label
        
    def getProb(self):
        return self._prob
        
    def getHist(self):
        return self._hist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def update(self,img,fgmask,bBox,frameNumber,hist,RoiResizeDim):

        matches = []
        
#        if bBox[0] <= 20 or bBox[0]+bBox[2] >= RoiResizeDim[0]-20 or bBox[1] <= 20 or bBox[1]+bBox[3] >= RoiResizeDim[1]-20 : #check if person1 is on edge of scene
#            boxOnEdge = True
#
#        else:
#            boxOnEdge = False


#        if len(matches) == 0: #new method 1
#            i = 0
#            for person in self.listOfPeople:
#                box1 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
#                lapping = overLap(box1,bBox) #largest overlap
#                if lapping > 0:
#                    histDist = histogramComparison(hist,person.hist)
#                    if len(matches)>0:
#                        if lapping >= matches[0][0]:
#                            if histDist < matches[0][3]:
#                                matches = [(lapping, i,0,histDist)]  #flag of one means it was found in  lost people
#                    else: #used in first iteration to set up overlap and histogram comparison
#                        matches = [(lapping, i,0,histDist)]
#                i = i + 1
        
        if len(matches) == 0: #new method 1
            i = 0
            for person in self.listOfPeople:
                box1 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
                lapping = overLap(box1,bBox) #largest overlap
                print (lapping, "lapping")
                #if lapping > 0:
                histDist = histogramComparison(hist,person.hist)
                print(histDist, "histDist")
                if histDist < 8000:
                    if len(matches)>0:
                        #if lapping >= matches[0][0]:
                        if histDist < matches[0][3]:
                            matches = [(lapping, i,0,histDist)]  #flag of one means it was found in  lost people
                    else: #used in first iteration to set up overlap and histogram comparison
                        matches = [(lapping, i,0,histDist)]
                i = i + 1


#        if len(matches) == 0 and boxOnEdge == False: #try to assign to person that is sharing a ROI
#            i = 0
#            for person in self.listOfPeople:
#                if person.sharedROI == True:
#                    p2 = Tools.pixelToWorld((person.fX+(person.fW/2),person.fY+person.fH), homography)
#                    dist = Tools.objectDistance(p1,p2)
#                    if dist < 5:
#                        histDist = Tools.histogramComparison(hist,person.hist)
#                    #print(histDist,'histDist 323')
#                        if len(matches)>0: #used after first iteration to compare overlap and histogram
#                        #if lapping > matches[0][0]:
#                            if histDist < matches[0][3]:
#                                matches = [([], i,0,histDist)]  #flag of one means it was found in  lost people
#                        else: #used in first iteration to set up overlap and histogram comparison
#                            matches = [([], i,0,histDist)]
#                i = i + 1
#            if len(matches) > 0:
#
#                person = self.listOfPeople[matches[0][1]]
#                if person.V > 0:
#                    person.roiCurrent = bBox
#                    person.lastGoodROI = bBox
#                    person.lastROICurrent = bBox
#                    person.fX,person.fY,person.fW,person.fH = person.roiCurrent[0],person.roiCurrent[1],person.roiCurrent[2],person.roiCurrent[3]
#                    person.V=0
#                    person.edgeCounter = 0
#                    person.roiCounter = 0
#                    #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 0',lostFlag,'lostFlag')
#                return


        if len(matches) == 0: #check lost people for match
            i = 0
            for person in self.lostListOfPeople:
                histDist = histogramComparison(hist,person.hist)
               # print(histDist,'histDist 323')
                if len(matches)>0: #used after first iteration to compare overlap and histogram
                    #if lapping > matches[0][0]:
                    if histDist < matches[0][3]:
                        matches = [([], i,1,histDist)]  #flag of one means it was found in  lost people
                else: #used in first iteration to set up overlap and histogram comparison
                    matches = [([], i,1,histDist)]
                i = i + 1
            if len(matches) > 0:
                if matches[0][3]> 7000: #hard coded value based on observations

                    matches = [] #histogram match is not close enough make  a new person
                    #lostFlag = 0
                else:
                    person = self.lostListOfPeople[matches[0][1]]
                    pointA = [person.location[-1][1],person.location[-1][2]]
                    pointB = [bBox[0]+(bBox[2]/2),bBox[1]+(bBox[3]/2)]
                    distM = objectDistance(pointA,pointB)
                    if distM < 50:
                        pass
                        #lostFlag = 1
                    else:
                        matches = []
                        #lostFlag = 0


        if len(matches)>0: #1 match found  update person attributes 
            flag = matches[0][2]
            index = matches[0][1]
            if flag == 0: #get the person from matches
                person = self.listOfPeople[index]
            else:
                person = self.lostListOfPeople[index]
                self.insertPerson(person,self.listOfPeople)
                self.removePerson(person.ID,self.lostListOfPeople)

            if person.V > 0:#frameNumber > person.location[-1][0]: #if this is the first hog box for a person in this frame update person
                if len(person.histList)< 5: #try to optimize the persons color histogram
            
                    
                    person.histList.append(hist)
                        #Tools.displayHistogram(person.hist,frameNumber,person.ID)
                    if len(person.histList) > 1: #optimize the histogram

                        #print(person.histList)
                        for hist in person.histList:
                            for i in range(len(person.hist)):
                                person.hist[i] = person.hist[i]+hist[i]
                        for i in range(len(person.hist)):
                            person.hist[i] = person.hist[i]/len(person.histList)


                person.V=0
                person.edgeCounter = 0
                person.roiCounter = 0
                #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 1',lostFlag,'lostFlag')
                return
            else:

                return




        elif len(matches) == 0:#4 no match found so create person after refining the hog box
           
            tmp_node=Person(self.index,bBox,0,hist) #step 3 only update persons histogram on creation, not in subsequent updates.
                                    
            self.listOfPeople.append(tmp_node)
            self.index=self.index+1
            #print('new person added roiCurrent cheated, index is '+ str(self.index) +' case 4e')
            return