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
        self.ROI_RESIZE_DIM_RGB = self.videoObjRGB.getSize()#(600,337)
        self.ROI_RESIZE_DIM_IR = self.videoObjRGB.getSize()#(600,337)
        
        
        self.resizeSetFlagIR = 0
        self.errorReportRGB = [0] #[numLabelmismatch,numFalsePos]
        self.errorReportIR = [0] #[numLabelmismatch,numFalsePos]
        self.pause = False
        self.svmDataRGB = []
        self.svmBoolsRGB = []
        self.svmDataIR = []
        self.svmBoolsIR = []
            
    def readAndTrack(self):
        #time1 = time.time()
        self.pause = False
        retRGB,imgRGB = self.capRGB.read()
        retIR,imgIR = self.capIR.read()
        
        if (not retRGB) or (not retIR): #allow for a graceful exit when the video ends
            print("Exiting Program End of Video...")
            print("MAXDIST " + str(MAXDIST))
            self.capRGB.release()
            self.capIR.release()
            cv2.destroyAllWindows()
            # train the SVM or write SVM dat to pickle files
                                    
            f = gzip.open( str(self.videoname)+"SVMDATARGB.pklz", "wb" )
            pickle.dump(self.svmDataRGB, f)
            f.close()
            f = gzip.open( str(self.videoname)+"SVMBOOLSRGB.pklz", "wb" )
            pickle.dump(self.svmBoolsRGB, f)
            f.close()
            f = gzip.open( str(self.videoname)+"SVMDATAIR.pklz", "wb" )
            pickle.dump(self.svmDataIR, f)
            f.close()
            f = gzip.open( str(self.videoname)+"SVMBOOLSIR.pklz", "wb" )
            pickle.dump(self.svmBoolsIR, f)
            f.close()
            
            
            
            
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
                    
                    
                    
                    currentMask = cv2.normalize(self.videoObjCurrentObjsRGB[i].getMask(), None, 0, 255, cv2.NORM_MINMAX)
                    tmpMask = currentMask.copy()
                    bBox = cv2.boundingRect(tmpMask)
                    print(bBox, "bBox in RGB read and track")
                    if self.videoObjCurrentObjsRGB[i].getLabel() == 'person' and (bBox[1]+bBox[3] > 300): #code for omitting background detections
                        #print(type(currentMask), "img channels")
                        hist = getHistRGBPYIMGSRCH(imgRGB,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM_RGB)
                        
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
            print("updating RGB people")
            trainingData, trainingBools = self.trackedPeopleRGB.update(imgRGB,self.frameNumber,detPersonList) 
                        
            if len(trainingData) > 0:
                self.svmDataRGB.extend(trainingData) #add the svm training data to master lists
                self.svmBoolsRGB.extend(trainingBools)
        else:
            print("Empty frame objects this frame")
        print("refreshing RGB people")
        self.trackedPeopleRGB.refresh(imgRGB,imgDisplayRGB,self.frameNumber,self.ROI_RESIZE_DIM_RGB) #update all of the people
        for person in self.trackedPeopleRGB.listOfPeople:
            if len(person.location) > 1:
                if objectDistance(person.location[-1],person.location[-2]) > 50:
                    self.errorReportRGB[0]+=1
                    self.pause = True
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
                   
                    currentMask = cv2.normalize(self.videoObjCurrentObjsIR[i].getMask(), None, 0, 255, cv2.NORM_MINMAX)
                    tmpMask = currentMask.copy()
                    bBox = cv2.boundingRect(tmpMask)
                    
                    if self.videoObjCurrentObjsIR[i].getLabel() == 'person' and (bBox[1]+bBox[3] > 300): #code for omitting background detections
                        
                        #print(type(currentMask), "img channels")
                        hist = getHistGray(imgIR,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM_IR)
                        #hist = getHistRGBPYIMGSRCH(imgIR,currentMask,0,0,currentMask.shape[1],currentMask.shape[0],self.ROI_RESIZE_DIM_IR)
                        tmpPerson = Detection(self.videoObjCurrentObjsIR[i].getMask(), bBox, self.videoObjCurrentObjsIR[i].getLabel(), self.videoObjCurrentObjsIR[i].getProb(), hist)
                        detPersonList.append(tmpPerson)
                        dispName = "IR"
                        displayHistogramGray(hist,dispName,self.frameNumber,i)
                        
                        
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
            print("updating IR people")
            trainingData, trainingBools = self.trackedPeopleIR.update(imgIR,self.frameNumber,detPersonList) 
            if len(trainingData) > 0:
                self.svmDataIR.extend(trainingData) #add the svm training data to master lists
                self.svmBoolsIR.extend(trainingBools)
        else:
            print("Empty frame objects this frame")
        print("refreshing IR people")
        self.trackedPeopleIR.refresh(imgIR,imgDisplayIR,self.frameNumber,self.ROI_RESIZE_DIM_IR) #update all of the people
        for person in self.trackedPeopleIR.listOfPeople:
            if len(person.location) > 1:
                if objectDistance(person.location[-1],person.location[-2]) > 50:
                    self.errorReportIR[0]+=1
                    self.pause = True
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
        
    
        print(self.errorReportRGB," RGB Labeling Errors")
        print(self.errorReportIR, " IR Labeling Errors")
        if (self.trackedPeopleRGB.index > 3):
            print("false positives count for RGB is ", self.trackedPeopleRGB.index-3 )
        else:
            print("false positives count for RGB is 0")
            
        if (self.trackedPeopleIR.index > 3):
            print("false positives count for IR is ", self.trackedPeopleIR.index-3 )
        else:
            print("false positives count for IR is 0")
        
        if (self.trackedPeopleRGB.index < 3):
            print("Number of untracked RGB people ", 3 - self.trackedPeopleRGB.index )
        else:
            print("Number of untracked RGB people = 0")
            
        if (self.trackedPeopleIR.index < 3):
            print("Number of untracked IR people ", 3 - self.trackedPeopleRGB.index )
        else:
            print("Number of untracked IR people = 0")
        
        print('framenumber ' + str(self.frameNumber))
        self.frameNumber += 1     
        k = cv2.waitKey(40) & 0xFF
        if k == ord('p'):
            print("Pausing...")
            return (None,2) #return 2 for paused
        elif k == ord('q'):
            print("Exiting Program...")
            self.capRGB.release()
            self.capIR.release()
            cv2.destroyAllWindows()
            return (None,0) #return 0 to toggle active off
        elif self.frameNumber-1 == 10120 or self.pause == True: #for testing only to pause at a certain frame
            #timeEnd = time.time()
            #totalTime = timeEnd - timeStart
            #print(totalTime,'totalTime')
#        elif self.pause : #for testing only to pause at a certain frame
            return (None,2)
        return (None,1) #return 1 to stay active

class People():
    ## The constructor.
    def __init__(self):
        self.listOfPeople=list()
        self.lostListOfPeople=list()
        self.index=0
        f = open( "svmModelRGB.p", "rb")
        self.SVMModelRGB = pickle.load(f)
        f.close()
        f = open( "svmModelRGB.p", "rb")
        self.SVMModelIR = pickle.load(f)
        f.close()
        #self.trackedPeople.update(img,self.fgmask,fX,fY,fW,fH,self.frameNumber,roi_hist,self.trackedPeople.listOfPeople)
        # Updates an item in the list of people/object or appends a new entry or assigns to a group or removes from a group
    def update(self,img,frameNumber,detectionsList): # update needs two passed in order to ensure that each person has a chance to match each box before the box is assigned to the wrond person
        trainingData = [] # stuff for SVM
        trainingBools = []
        if len(detectionsList) == 0: #in case method is called on a empty detectionsList
            return trainingData, trainingBools
        personList = self.listOfPeople
        
        if len(personList) == 0: # if length of person list is zero generate a person for each detection
            i = 0
            for detection in detectionsList:
                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist(),detection.getMask()) #step 3 only update persons histogram on creation, not in subsequent updates.
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                print("creating person for the first time!!",self.index-1)
                i += 1
            
        
        else: 
            matchesList = [] # the same length as personList  
            for i in range(len(personList)): # generate the list of matches 
                bestMatch =  [-1, i, -1, 1000000] #[pixelDist, personIdx, detectionIdx,histDist= 1000000 majic number should be replaced with better alternative]
                person = personList[i]
                for j in range(len(detectionsList)):
                    detection = detectionsList[j]
                        
                    histDist = histogramComparison(detectionsList[j].getHist(),personList[i].hist)
                    print(histDist, "histDist for person ",personList[i].ID, " and detection ", j)
                    #pixelDist = objectDistance(getCentroid(detectionsList[j].getMask()),getCentroid(personList[i].mask))
                    box0 = detection.getBbox()
                    
                    p1 = [box0[0]+box0[2]/2,box0[1]+box0[3]/2]
                    p2 = [person.fX + person.fW/2,person.fY+person.fH/2]
                    pixelDist = objectDistance(p1,p2)
                    if pixelDist < 75:

                        if bestMatch[2] == -1:#j == 0:
                            bestMatch = [pixelDist, i, j, histDist]  # first iteration do not save to training data yet
                        else:
                            if histDist > bestMatch[3] : #for some histograms larger value is better 0,2 Larger is better 1,3,4,,5 SMALLER is better
                                trainingData.append([bestMatch[3],bestMatch[0]]) # save previous hist and pixel dist as a vector in training data when histogram match for current is better than previous
                                trainingBools.append(0) # save classification for negative to bool list
                                bestMatch = [pixelDist, i, j, histDist] 
                            else:
                                trainingData.append([histDist,pixelDist]) # save hist and pixel dist as a vector in training data when hist distance is not better than the current best match
                                trainingBools.append(0) # save classification for negative to bool list
                    else: # put data into list as negative
                        trainingData.append([histDist,pixelDist]) # save hist and pixel dist as a vector in training data when pixel distance is not satisfied
                        trainingBools.append(0) # save classification for negative to bool list
                matchesList.append(bestMatch) #save the match for use in second pass
            print(matchesList, " matchesList")
            #del bestMatch
            usedDetections = []
            boolListPerson = []
            
            for i in range(len(personList)):  #assign detections from results of first pass 
                if matchesList[i][2] == -1:
                    continue # no match found
                match = i # number to use to compare against i in order to see if the match was correct
                for j in range(len(personList)):
                    if matchesList[i][2] == matchesList[j][2] and i != j: # check if there is a disagreement on the match and ensure that we are not talking about the same person
                        if matchesList[j][3] > matchesList[i][3]: #compare hist dists to see who had the better match !! the equality needs to match the one above
                            match = j #give the match to the other guy
                    
                if match == i: #match was correct update person and update list that shows which detections were used
                    personList[i].V=0
                    personList[i].updateLocation(detectionsList[matchesList[i][2]].getBbox())
                    personList[i].mask = detectionsList[matchesList[i][2]].getMask()
                    if frameNumber % 1 == 0:
                        personList[i].hist = detectionsList[matchesList[i][2]].getHist()
                    print("updating person ", personList[i].ID, " with detection # ", matchesList[i][2])
                    
                    usedDetections.append(matchesList[i][2])
                    boolListPerson.append(True)
                    trainingData.append([matchesList[i][3],matchesList[i][0]]) # save hist and pixel dist as a vector in training data when match was valid. Only do this for videos that have been manually verified to not have errors in tracking!!!
                    trainingBools.append(1) # save classification for negative to bool list
                    
                else:
                    boolListPerson.append(False)
                    #might need code here to chose other detection if pixel proximity is ok
                    print("did not find detection for person ", i)
                    trainingData.append([matchesList[i][3],matchesList[i][0]]) # save hist and pixel dist as a vector in training data when match was invalid
                    trainingBools.append(0) # save classification for negative to bool list
            
#            for i in range(len(boolListPerson)): #second chance 
#                if boolListPerson[i] == False and personList[i].V < 10 and personList[i].nearEdge == False: #person is still present in scene do not spawn new person.
#                    firstCompare = True
#                    for j in range(len(detectionsList)): #spawn new person for the remaining detections
#                        if j in usedDetections:
#                            pass
#                        else:
#                            histDist = histogramComparison(detectionsList[j].getHist(),personList[i].hist)
#                            print(histDist, "histDist for person ",personList[i].ID, " and detection second chance", j)
#                            if firstCompare == True:
#                                bestMatch = [-1, i, j, histDist]  
#                                firstCompare = False
#                            else:
#                                if histDist > bestMatch[3] : #for some histograms larger value is better 0,2 Larger is better 1,3,4,,5 SMALLER is better
#                                    bestMatch = [-1, i, j, histDist] 
#                    matchesList[i] = bestMatch #update matches list
#            for i in range(len(boolListPerson)):
#                if boolListPerson[i] == False and personList[i].V < 10 and personList[i].nearEdge == False: #person is still present in scene do not spawn new person
#                    match = i
#                    for j in range(len(boolListPerson)):
#                        if boolListPerson[j] == False and personList[j].V < 10 and personList[j].nearEdge == False:
#                            if matchesList[i][2] == matchesList[j][2] and i != j: # check if there is a disagreement on the match and ensure that we arenot talking about the same person
#                                if matchesList[j][3] > matchesList[i][3]: #compare hist dists to see who had the better match !! the equality needs to match the one above
#                                    match = j #give the match to the other guy    
#                if match == i: #match was correct update person and update list that shows which detections were used
#                    personList[i].V=0
#                    personList[i].updateLocation(detectionsList[matchesList[i][2]].getBbox())
#                    personList[i].mask = detectionsList[matchesList[i][2]].getMask()
#                    if frameNumber % 1 == 0:
#                        personList[i].hist = detectionsList[matchesList[i][2]].getHist()
#                    print("updating person ", personList[i].ID, " with detection # in second chance", matchesList[i][2])
#                    
#                    usedDetections.append(matchesList[i][2])
#                    boolListPerson[i] =True
#                    
#                else:
#                    boolListPerson[i] = False
#                    #might need code here to chose other detection if pixel proximity is ok
#                    print("did not find detection for person in second chance ", personList[i].ID)
#                    
            for i in range(len(boolListPerson)): #second chanceversion 2
                if boolListPerson[i] == False and personList[i].nearEdge == False: #person is still present in scene do not spawn new person.
                    for j in range(len(detectionsList)): #spawn new person for the remaining detections
                        if j in usedDetections:
                            pass
                        else:
                            personList[i].V=0
                            personList[i].updateLocation(detectionsList[j].getBbox())
                            personList[i].mask = detectionsList[j].getMask()
                            if frameNumber % 1 == 0:
                                personList[i].hist = detectionsList[j].getHist()
                            print("updating person ", personList[i].ID, " with detection # in second chance", j)
                            
                            usedDetections.append(j)
                            boolListPerson[i] =True
                    










        
                        
            for j in range(len(detectionsList)): #spawn new person for the remaining detections
                if j in usedDetections:
                    pass
                else:
                    tmp_node=Person(self.index,detectionsList[j].getBbox(),0,detectionsList[j].getHist(),detectionsList[j].getMask()) 
                    self.listOfPeople.append(tmp_node)
                    print("creating NEW person !!",self.index)
                    self.index=self.index+1
        return trainingData, trainingBools  
            
#            i = 0 #original method which did not have a second pass
#            for person in personList: #iterate through the people to attempte to find a match twice
#                print("attempting to find match for person ", person.ID)
#                bestMatch =  [-1, i, -1, 1000000]
#                j = 0
#                for detection in detectionsList:
#                    
#                    box0 = detection.getBbox()
#                    box1 = [box0[0],box0[1],box0[0]+box0[2],box0[1]+box0[3]]
#                    box2 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
#                    lapping = overLap(box1,box2) 
#                    p1 = [box0[0]+box0[2]/2,box0[1]+box0[3]/2]
#                    p2 = [person.fX + person.fW/2,person.fY+person.fH/2]
#                    #pixelDist = objectDistance(p1,p2)
#                    pixelDist = objectDistance(getCentroid(detection.getMask()),getCentroid(person.mask))
#                    mask = detection.getMask()
#                    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
#                    #print(mask.shape)
#                    
#                                    
#                    
#                    print(pixelDist, " Pixel Dist")
#                    #print (lapping, "lapping")
#                    histDist = histogramComparison(detection.getHist(),person.hist)
#                    print(histDist, "histDist for person ",person.ID, " and detection ", j)
 #                    pixelDist = objectDistance(getCentroid(detection.getMask()),getCentroid(person.mask))
#                    if pixelDist < 75:
#                        if histDist < bestMatch[3] :
#                            bestMatch = [pixelDist, i, j, histDist]  
#                    
#                    j = j + 1
#            
#                if bestMatch[2] != -1:
#                    global MAXDIST
#                    if bestMatch[3] > MAXDIST:
#                        MAXDIST = bestMatch[3]
#                    person.V=0
#                    person.updateLocation(detectionsList[bestMatch[2]].getBbox())
#                    person.mask = detectionsList[bestMatch[2]].getMask()
#                    if frameNumber % 30 == 0:
#                        person.hist = detectionsList[bestMatch[2]].getHist()
#                    print("updating person ", person.ID, " with detection # ", bestMatch[2])
#                    del detectionsList[bestMatch[2]]
#                else:
#                    print("did not find detection for person ", i)
#                i = i + 1
#           
#            for detection in detectionsList: #spawn new person for the remaining detections
#                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist(),detection.getMask()) 
#                self.listOfPeople.append(tmp_node)
#                print("creating NEW person !!",self.index)
#                self.index=self.index+1

    def updateWithSVM(self,img,frameNumber,detectionsList,imageType): # update needs two passed in order to ensure that each person has a chance to match each box before the box is assigned to the wrond person
        

        
        if len(detectionsList) == 0: #in case method is called on a empty detectionsList
            return
        personList = self.listOfPeople
        if imageType == "RGB":
            model = self.SVMModelRGB
        elif imageType == "IR":
            model = self.SVMModelIR
        if len(personList) == 0: # if length of person list is zero generate a person for each detection
            i = 0
            for detection in detectionsList:
                tmp_node=Person(self.index,detection.getBbox(),0,detection.getHist(),detection.getMask()) #step 3 only update persons histogram on creation, not in subsequent updates.
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                print("creating person for the first time!!",self.index-1)
                i += 1
            
        
        else: 
            matchesList = [] # the same length as personList  
            for i in range(len(personList)): # generate the list of matches 
                bestMatch =  [-1, i, -1, 1000000] #[pixelDist, personIdx, detectionIdx,histDist= 1000000 majic number should be replaced with better alternative]
                person = personList[i]
                for j in range(len(detectionsList)):
                    detection = detectionsList[j]
                        
                    histDist = histogramComparison(detectionsList[j].getHist(),personList[i].hist)
                    print(histDist, "histDist for person ",personList[i].ID, " and detection ", j)
                    #pixelDist = objectDistance(getCentroid(detectionsList[j].getMask()),getCentroid(personList[i].mask))
                    box0 = detection.getBbox()
                    
                    p1 = [box0[0]+box0[2]/2,box0[1]+box0[3]/2]
                    p2 = [person.fX + person.fW/2,person.fY+person.fH/2]
                    pixelDist = objectDistance(p1,p2)
                    if pixelDist < 75:

                        if bestMatch[2] == -1:#j == 0:
                            bestMatch = [pixelDist, i, j, histDist]  # first iteration do not save to training data yet
                        else:
                            if histDist > bestMatch[3] : #for some histograms larger value is better 0,2 Larger is better 1,3,4,,5 SMALLER is better
                                
                                bestMatch = [pixelDist, i, j, histDist] 
                            
                    
                matchesList.append(bestMatch) #save the match for use in second pass
            print(matchesList, " matchesList")
            #del bestMatch
            usedDetections = []
            boolListPerson = []
            
            for i in range(len(personList)):  #assign detections from results of first pass 
                if matchesList[i][2] == -1:
                    continue # no match found
                match = i # number to use to compare against i in order to see if the match was correct
                for j in range(len(personList)):
                    if matchesList[i][2] == matchesList[j][2] and i != j: # check if there is a disagreement on the match and ensure that we are not talking about the same person
                        if matchesList[j][3] > matchesList[i][3]: #compare hist dists to see who had the better match !! the equality needs to match the one above
                            match = j #give the match to the other guy
                    
                if match == i: #match was correct update person and update list that shows which detections were used
                    personList[i].V=0
                    personList[i].updateLocation(detectionsList[matchesList[i][2]].getBbox())
                    personList[i].mask = detectionsList[matchesList[i][2]].getMask()
                    if frameNumber % 1 == 0:
                        personList[i].hist = detectionsList[matchesList[i][2]].getHist()
                    print("updating person ", personList[i].ID, " with detection # ", matchesList[i][2])
                    
                    usedDetections.append(matchesList[i][2])
                    boolListPerson.append(True)
                    
                    
                else:
                    boolListPerson.append(False)
                    #might need code here to chose other detection if pixel proximity is ok
                    print("did not find detection for person ", i)
                    
        
            for i in range(len(boolListPerson)): #second chanceversion 2
                if boolListPerson[i] == False and personList[i].nearEdge == False: #person is still present in scene do not spawn new person.
                    for j in range(len(detectionsList)): #spawn new person for the remaining detections
                        if j in usedDetections:
                            pass
                        else:
                            personList[i].V=0
                            personList[i].updateLocation(detectionsList[j].getBbox())
                            personList[i].mask = detectionsList[j].getMask()
                            if frameNumber % 1 == 0:
                                personList[i].hist = detectionsList[j].getHist()
                            print("updating person ", personList[i].ID, " with detection # in second chance", j)
                            
                            usedDetections.append(j)
                            boolListPerson[i] =True
                    










        
                        
            for j in range(len(detectionsList)): #spawn new person for the remaining detections
                if j in usedDetections:
                    pass
                else:
                    tmp_node=Person(self.index,detectionsList[j].getBbox(),0,detectionsList[j].getHist(),detectionsList[j].getMask()) 
                    self.listOfPeople.append(tmp_node)
                    print("creating NEW person !!",self.index)
                    self.index=self.index+1
        return
            

    def refresh(self,img,imgCopy,frameNumber,RoiResizeDim): #updates people's boxes and checks for occlusion
        personList = list(self.listOfPeople) #make copy of people list to use for while loop

        while len(personList) > 0:
            

            person1 = self.getPerson(personList[0].ID,self.listOfPeople)
            #print(person1.V, "person visibility in refresh", person1.ID)
            #print(person1.nearEdge, "person visibility in refresh", person1.ID)
            #if (person1.V > 10) or (person1.V > 1 and person1.nearEdge == True):
            
            #print(person1.ID,'moving = ', person1.moving)
            #flag = 0
            
            if (person1.fX < 40) or (person1.fX+ person1.fW > RoiResizeDim[0] - 10):
                person1.nearEdge = True
            
            if (person1.V > 15):
                self.removePerson(person1.ID,self.listOfPeople)
            elif (person1.V >= 10 and person1.nearEdge == True):
                self.removePerson(person1.ID,self.listOfPeople)
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
            print(person1.V, "person visibility in refresh", person1.ID)
            print(person1.nearEdge, "person nearEdge", person1.ID)
            person1.V=person1.V+1
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

    def __init__(self,ID,bBox,visible,hist,mask):
        self.ID=ID
        self.fX=bBox[0]
        self.fY=bBox[1]
        self.fW=bBox[2]
        self.fH=bBox[3]
        self.centroid = getCentroid(mask)
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
        self.mask = mask
    
    def updateLocation(self, bBox):
        self.fX=bBox[0]
        self.fY=bBox[1]
        self.fW=bBox[2]
        self.fH=bBox[3]

def displayHistogram(histogram,dispName,frameNumber=-1,id=-1):
    print("display testRGB!!!!!!!!#######")
    print(histogram.shape)
    histogram = histogram.reshape(-1)
    
    binCount = histogram.shape[0]
    print(histogram.shape)
    BIN_WIDTH = 1#3
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

def displayHistogramGray(histogram,dispName,frameNumber=-1,id=-1):
    print("display testIR!!!!!!!!#######")
    print(histogram.shape)
    histogram = histogram.reshape(-1)
    print(histogram.shape)
    binCount = histogram.shape[0]
    BIN_WIDTH = 3
    img = np.zeros((256, binCount*BIN_WIDTH, 3), np.uint8)
    for i in xrange(binCount):
        h = int(histogram[i])
        cv2.rectangle(img, (i*BIN_WIDTH, 255), ((i+1)*BIN_WIDTH, 255-h), (i, i, i), -1)
        cv2.rectangle(img, (i*BIN_WIDTH, 255-h), ((i+1)*BIN_WIDTH, 0), (255-i, 255-i, 255-i), -1)
    #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if(frameNumber != -1):
        cv2.putText(img,"Mask_"+str(id)+" Histogram",(10,20),0, .75, (255,255,255), 1,8, False)
    if(id!=-1):
        cv2.imshow("Hist_"+str(id)+dispName, cv2.resize(img,(img.shape[1]/2,img.shape[0]/3)))
    else:
        cv2.imshow("Probable Person Histogram", img)

def getHistRGB(img,mask,fX,fY,fW,fH,ROI_RESIZE_DIM): #not a foreground hist
 
    hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv_roi = img
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])   
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        
    return roi_hist

def getHistRGBPYIMGSRCH(img,mask,fX,fY,fW,fH,ROI_RESIZE_DIM):
    
    hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #cv2.normalize(hist,hist).flatten()
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX).flatten()
    return hist

def getHistGray(img,mask,fX,fY,fW,fH,ROI_RESIZE_DIM): #seems to work better for grey scale images
 
    #hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_roi = img
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])   
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        
    return roi_hist


    
    
def histogramComparison(curHist,newHist):
    
    
    distance = cv2.compareHist(curHist,newHist,0) #update based on color match 4 Bhattacharya distance, 0 and 2,3 work well, 1,4 and 5 do not , correlation seems to work best for RGB and IR
    #cv2.HISTCMP_BHATTACHARYYA = 3 smaller is better
    #cv2.HISTCMP_CHISQR = 1 smaller is better
    #cv2.HISTCMP_CHISQR_ALT = 4 smaller is better
    #cv2.HISTCMP_CORREL = 0 larger is better
    #cv2.HISTCMP_HELLINGER = 3 smaller is better
    #cv2.HISTCMP_INTERSECT = 2 larger is better
    #cv2.HISTCMP_KL_DIV = 5 smaller is better? unsure
    
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
    
def getCentroid(mask):
    mom = cv2.moments(mask)
    #mom = cv2.HuMoments(mom) not sure how to use humoments yet
    moment10 = mom["m10"];
    moment01 = mom["m01"];
    moment00 = mom["m00"];
    x = moment10 / moment00;
    y = moment01 / moment00;
    #print("centroid is ", [x,y])
    return (x,y)
