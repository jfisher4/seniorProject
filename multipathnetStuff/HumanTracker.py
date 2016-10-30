import cv2
import numpy as np
#import lutorpy as lua
import pickle
import gzip
#import time
from storage import *
#import os

global MAXDIST
MAXDIST = 0.0
class HumanTracker:

    def __init__(self, directory, videoname):
        self.directory = directory
        self.videoname = videoname
        self.metadata = videoname.split(".")
        print(self.metadata, " test")
        self.capRGB = cv2.VideoCapture(directory+self.metadata[0]+"_RGB."+self.metadata[1])
        self.capIR = cv2.VideoCapture(directory+self.metadata[0]+"_IR."+self.metadata[1])
        self.frameNumber = 0
        f = gzip.open( directory+self.metadata[0]+"_RGB."+self.metadata[1]+".pklz", "rb" )
        self.videoObjRGB = pickle.load(f)
        f.close()
        f = gzip.open( directory+self.metadata[0]+"_IR."+self.metadata[1]+".pklz", "rb" )
        self.videoObjIR = pickle.load(f)
        f.close()
        print(len(self.videoObjRGB.getFrames()), 'length of get frames of restored video object')
        print(len(self.videoObjIR.getFrames()), 'length of get frames of restored video object')
        self.videoObjCurrentObjsRGB = None
        self.videoObjCurrentObjsIR = None
        self.trackedPeopleRGB = People()
        self.trackedPeopleIR = People()
        self.ROI_RESIZE_DIM_RGB = (600,337)
        self.ROI_RESIZE_DIM_IR = (600,337)
        self.resizeSetFlagRGB = 0
        self.resizeSetFlagIR = 0
            
    def readAndTrack(self):
        #time1 = time.time()
        retRGB,imgRGB = self.capRGB.read()
        retIR,imgIR = self.capIR.read()
        if (not retRGB) or (not retIR): #allow for a graceful exit when the video ends
            print("Exiting Program End of Video...")
            print("MAXDIST " + str(MAXDIST))
            self.capRGB.release()
            self.capIR.release()
            cv2.destroyAllWindows()
            return(None, 0) #return 0 to toggle active off
        imgRGB = cv2.resize(imgRGB,self.ROI_RESIZE_DIM_RGB)
        imgDisplayRGB = imgRGB.copy()
        imgIR = cv2.resize(imgIR,self.ROI_RESIZE_DIM_IR)
        imgDisplayIR = imgIR.copy()
        self.videoObjCurrentObjsRGB = self.videoObjRGB.getFrames()[self.frameNumber].getImageObjects()
        self.videoObjCurrentObjsIR = self.videoObjIR.getFrames()[self.frameNumber].getImageObjects()
        if self.videoObjCurrentObjsRGB[0].getMask() != None: # for some reason there exists some none objects in the frames image object list. 
            detPersonList = []
            for i in range(len(self.videoObjCurrentObjsRGB)):
                if self.videoObjCurrentObjsRGB[i].getLabel() != None:
                    #print(self.videoObjCurrentObjsRGB[i].getLabel())
                    
                    if self.resizeSetFlagRGB == 0: #only do this one time
                        self.ROI_RESIZE_DIM_RGB = (self.videoObjCurrentObjsRGB[i].getMask().shape[1],self.videoObjCurrentObjsRGB[i].getMask().shape[0])
                        imgRGB = cv2.resize(imgRGB,self.ROI_RESIZE_DIM_RGB)
                        self.resizeSetFlagRGB = 1
                    
                    currentMask = cv2.normalize(self.videoObjCurrentObjsRGB[i].getMask(), None, 0, 255, cv2.NORM_MINMAX)
                    tmpMask = currentMask.copy()
                    bBox = cv2.boundingRect(tmpMask)
                    
                    if self.videoObjCurrentObjsRGB[i].getLabel() == 'person':
                        #print(type(currentMask), "img channels")
                        hist = getHist(imgRGB,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM_RGB)
                        
                        tmpPerson = Detection(self.videoObjCurrentObjsRGB[i].getMask(), bBox, self.videoObjCurrentObjsRGB[i].getLabel(), self.videoObjCurrentObjsRGB[i].getProb(), hist)
                        detPersonList.append(tmpPerson)
                        dispName = "RGB"
                        displayHistogram(hist,dispName,self.frameNumber,i)
                        
                        cv2.rectangle(imgDisplayRGB, (bBox[0], bBox[1]), (bBox[0]+bBox[2],bBox[1]+bBox[3]), (255,0,0), 4)
                        
                        #bBox = newBox
                        #print(bBox,"bBox")
                        
                                        
                    else:
                        print(self.videoObjCurrentObjsRGB[i].getLabel()) 
                        
                    #bBox = self.videoObjCurrentObjsIR[i].getBbox()
                    bBox = cv2.boundingRect(currentMask)
                    printString = 'Frame Num' + str(self.frameNumber)
                    cv2.putText(currentMask,printString,(10,20),0,.5, (255,255,255),1,8,False)
                    #print(bBox)
                    #bBox = bBox.astype(int)
                    #bBox = [bBox[0],bBox[1],bBox[2]-bBox[0],bBox[3]-bBox[1]] #convert from x1,y1,x2,y2 to x,y,w,h
                    #cv2.rectangle(currentMask, (bBox[0], bBox[1]), (bBox[0]+bBox[2],bBox[1]+bBox[3]), (255,0,0), 4)
                    cv2.imshow("Mask_RGB_"+str(i),cv2.resize(currentMask,(currentMask.shape[1]/2,currentMask.shape[0]/2)))
                    #people class stuff
                                                                          
                    
                else:
                    print(self.videoObjCurrentObjsRGB[i].getLabel()," the label of problem object")
                    #print(type(self.videoObjCurrentObjsRGB[i].getMask()))
            self.trackedPeopleRGB.update(imgRGB,self.frameNumber,detPersonList) 
        else:
            print("Empty frame objects this frame")
        
        self.trackedPeopleRGB.refresh(imgRGB,imgDisplayRGB,self.frameNumber,self.ROI_RESIZE_DIM_RGB) #update all of the people
        for person in self.trackedPeopleRGB.listOfPeople:
            if person.V == 1:# HOG has updated visibility this frame
                
                cv2.rectangle(imgDisplayRGB, (person.fX, person.fY), (person.fX+person.fW,person.fY+person.fH), (0,0,255), 2)
                cv2.putText(imgDisplayRGB,"Person " + str(person.ID),(person.fX+5,person.fY-20),0, .5, (0,0,255), 2,8, False)
            elif person.V < 1000: #HOG did not update visibility and person was tracked with background subtracction
                cv2.rectangle(imgDisplayRGB, (person.fX, person.fY), (person.fX+person.fW, person.fY+person.fH), (0,255,0), 2) #show meanshift roi box green
                cv2.putText(imgDisplayRGB,"Person " + str(person.ID),(person.fX+5,person.fY-20),0, .5, (0,255,0), 2,8, False)
        

            
        height, width = imgRGB.shape[:2]
        printString = 'Frame RGB' + str(self.frameNumber)
        cv2.putText(imgDisplayRGB,printString,(10,20),0,.5, (0,0,255),1,8,False)
        cv2.imshow(self.metadata[0] + "RGB",imgDisplayRGB) 
        #cv2.imshow("hsv",cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV))
        
        if self.videoObjCurrentObjsIR[0].getMask() != None: # for some reason there exists some none objects in the frames image object list. 
            detPersonList = []
            for i in range(len(self.videoObjCurrentObjsIR)):
                if self.videoObjCurrentObjsIR[i].getLabel() != None:
                    #print(self.videoObjCurrentObjsIR[i].getLabel())
                    
                    if self.resizeSetFlagIR == 0: #only do this one time
                        self.ROI_RESIZE_DIM_IR = (self.videoObjCurrentObjsIR[i].getMask().shape[1],self.videoObjCurrentObjsIR[i].getMask().shape[0])
                        imgIR = cv2.resize(imgIR,self.ROI_RESIZE_DIM_IR)
                        self.resizeSetFlagIR = 1
                    
                    currentMask = cv2.normalize(self.videoObjCurrentObjsIR[i].getMask(), None, 0, 255, cv2.NORM_MINMAX)
                    tmpMask = currentMask.copy()
                    bBox = cv2.boundingRect(tmpMask)
                    
                    if self.videoObjCurrentObjsIR[i].getLabel() == 'person':
                        #print(type(currentMask), "img channels")
                        hist = getHist(imgIR,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM_IR)
                        
                        tmpPerson = Detection(self.videoObjCurrentObjsIR[i].getMask(), bBox, self.videoObjCurrentObjsIR[i].getLabel(), self.videoObjCurrentObjsIR[i].getProb(), hist)
                        detPersonList.append(tmpPerson)
                        dispName = "IR"
                        displayHistogram(hist,dispName,self.frameNumber,i)
                        
                        
                        cv2.rectangle(imgDisplayIR, (bBox[0], bBox[1]), (bBox[0]+bBox[2],bBox[1]+bBox[3]), (255,0,0), 4)
                        
                        #bBox = newBox
                        #print(bBox,"bBox")
                        
                                        
                    else:
                        print(self.videoObjCurrentObjsIR[i].getLabel()) 
                    
                    #bBox = self.videoObjCurrentObjsIR[i].getBbox()
                    bBox = cv2.boundingRect(currentMask)
                    printString = 'Frame Num' + str(self.frameNumber)
                    cv2.putText(currentMask,printString,(10,20),0,.5, (255,255,255),1,8,False)
                    #print(bBox)
                    #bBox = bBox.astype(int)
                    #bBox = [bBox[0],bBox[1],bBox[2]-bBox[0],bBox[3]-bBox[1]] #convert from x1,y1,x2,y2 to x,y,w,h
                    #cv2.rectangle(currentMask, (bBox[0], bBox[1]), (bBox[0]+bBox[2],bBox[1]+bBox[3]), (255,0,0), 4)
                    cv2.imshow("Mask_IR_"+str(i),cv2.resize(currentMask,(currentMask.shape[1]/2,currentMask.shape[0]/2)))
                    #people class stuff
                                                                          
                    
                else:
                    print(self.videoObjCurrentObjsIR[i].getLabel()," the label of problem object")
                    #print(type(self.videoObjCurrentObjsIR[i].getMask()))
            self.trackedPeopleIR.update(imgIR,self.frameNumber,detPersonList) 
        else:
            print("Empty frame objects this frame")
        
        self.trackedPeopleIR.refresh(imgIR,imgDisplayIR,self.frameNumber,self.ROI_RESIZE_DIM_IR) #update all of the people
        for person in self.trackedPeopleIR.listOfPeople:
            if person.V == 1:# HOG has updated visibility this frame
                
                cv2.rectangle(imgDisplayIR, (person.fX, person.fY), (person.fX+person.fW,person.fY+person.fH), (0,0,255), 2)
                cv2.putText(imgDisplayIR,"Person " + str(person.ID),(person.fX+5,person.fY-20),0, .5, (0,0,255), 2,8, False)
            elif person.V < 1000: #HOG did not update visibility and person was tracked with background subtracction
                cv2.rectangle(imgDisplayIR, (person.fX, person.fY), (person.fX+person.fW, person.fY+person.fH), (0,255,0), 2) #show meanshift roi box green
                cv2.putText(imgDisplayIR,"Person " + str(person.ID),(person.fX+5,person.fY-20),0, .5, (0,255,0), 2,8, False)
        

            
        height, width = imgIR.shape[:2]
        printString = 'Frame IR' + str(self.frameNumber)
        cv2.putText(imgDisplayIR,printString,(10,20),0,.5, (0,0,255),1,8,False)
        cv2.imshow(self.metadata[0] + "IR",imgDisplayIR) 
        #cv2.imshow("hsv",cv2.cvtColor(imgIR, cv2.COLOR_BGR2HSV))
        
    
        
        
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
        elif self.frameNumber == 100: #for testing only to pause at a certain frame
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
                print("creating person for the first time!!",self.index)
                i += 1
       
        else:
            i = 0
            for person in personList: #iterate through the people to attempte to find a match
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
                        #if lapping > 0.0 or pixelDist < 300:
                        if pixelDist < 100:
                            if histDist < bestMatch[3] :
                                bestMatch = [lapping, i, j, histDist]  
                    else: #used in first iteration to set up overlap and histogram comparison
                        #if lapping > 0.0 or pixelDist < 300:
                        if pixelDist < 100:
                            bestMatch = [lapping, i, j, histDist]
                    j = j + 1
            
                if len(bestMatch)>0:
                    global MAXDIST
                    if bestMatch[3] > MAXDIST:
                        MAXDIST = bestMatch[3]
                    person.V=0
                    person.updateLocation(detectionsList[bestMatch[2]].getBbox())
                    person.hist = detectionsList[bestMatch[2]].getHist()
                    print("updating person ", person.ID)
                    del detectionsList[bestMatch[2]]
                else:
                    print("did not find detection for person ", i)
                i = i + 1
           
            for detection in detectionsList: #spawn new person for the remaining detections
                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist()) 
                self.listOfPeople.append(tmp_node)
                print("creating NEW person !!",self.index)
                self.index=self.index+1
                
   

    def refresh(self,img,imgCopy,frameNumber,RoiResizeDim): #updates people's boxes and checks for occlusion
        personList = list(self.listOfPeople) #make copy of people list to use for while loop

        while len(personList) > 0:
            

            person1 = self.getPerson(personList[0].ID,self.listOfPeople)
            if person1.V > 30:
                self.removePerson(person1.ID,self.listOfPeople)
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

def displayHistogram(histogram,dispName,frameNumber=-1,id=-1):
    histogram = histogram.reshape(-1)
    binCount = histogram.shape[0]
    BIN_WIDTH = 3
    img = np.zeros((256, binCount*BIN_WIDTH, 3), np.uint8)
    for i in xrange(binCount):
        h = int(histogram[i])
        cv2.rectangle(img, (i*BIN_WIDTH+1, 255), ((i+1)*BIN_WIDTH-1, 255-h), (int(180.0*i/binCount), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if(frameNumber != -1):
        cv2.putText(img,"Mask_"+str(id)+" Histogram",(10,20),0, .75, (255,255,255), 1,8, False)
    if(id!=-1):
        cv2.imshow("Hist_"+str(id)+dispName, cv2.resize(img,(img.shape[1]/2,img.shape[0]/3)))
    else:
        cv2.imshow("Probable Person Histogram", img)

def getHist(img,mask,fX,fY,fW,fH,ROI_RESIZE_DIM): #not a foreground hist
 
    #hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_roi = img
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
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
    
