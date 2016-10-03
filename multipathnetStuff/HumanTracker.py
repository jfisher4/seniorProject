import cv2
import numpy as np
import lutorpy as lua
import pickle
#import the lua script
require("demo")
ROI_RESIZE_DIM = (600,400)
lua.LuaRuntime(zero_based_index=False)

class HumanTracker:
    
    def __init__(self, directory, videoname):
        self.directory = directory
        self.cap = cv2.VideoCapture(videoname)
        self.metadata = videoname.split("_")
        #self.homography = pickle.load( open( self.metadata[0]+"_H.p", "rb" ) )
        #self.rotationMatrix = pickle.load(open( self.metadata[0]+"_rotM.p","rb"))
        #self.cameraPosition = Tools.cameraPosition(self.rotationMatrix, pickle.load( open( self.metadata[0]+"_tvec.p", "rb" ) ))
        self.frameNumber = 0
        self.saveName=directory + "../multipathnet/deepmask/data/testTmp.jpeg" # save the file so that we can load it into lua, hopefully we can find a better way
        self.multiPathObject = MultiPathNet
        #init the object
        multiPathObject.init(multiPathObject)
        multiPathObject.start(multiPathObject)
        self.trackedPeople = People()
        _, self.prvs = self.cap.read()
	
    def readAndTrack(self):
        time1 = time.time()
        ret,img = self.cap.read()
        if not ret: #allow for a graceful exit when the video ends
            print("Exiting Program End of Video...")
            self.cap.release()
            Tools.cv2.destroyAllWindows()
            return(None, 0) #return 0 to toggle active off

        self.frameNumber += 1
        luaImg = np.reshape(img, (img.shape[2],img.shape[0],img.shape[1]))#convert the shape to the same as lua load image, but there is still somethign wrong with the data
        luaImg = torch.fromNumpyArray(luaImg)
        cv2.imwrite(self.saveName, luaImg)
        #print('framenumber ' + str(self.frameNumber))
        probs, names, masks = multiPathObject.processImg(multiPathObject,luaImg)
        #print('W', WIDTH, 'H', HEIGHT)
        masks = masks.asNumpyArray()
        for maskNum in masks.shape[0]:
            if labels[maskNum] == "person":
                currentMask = masks[maskNum].reshape(masks.shape[1],masks.shape[2]) # convert to uint8 array of the same dim as the image
                #get color histogram from mask

                #match to existing person or create new person and give attribute mask to the person object
            cv2.imshow('Masks'+str(maskNum),currentMask)
        imgDisplay = img.copy()
        k = Tools.cv2.waitKey(2) & 0xFF
        if k == ord('p'):
            print("Pausing...")
            return (None,2) #return 2 for paused
        elif k == ord('q'):
            print("Exiting Program...")
            self.cap.release()
            Tools.cv2.destroyAllWindows()
            return (None,0) #return 0 to toggle active off
        elif self.frameNumber == 10000: #for testing only to pause at a certain frame
            timeEnd = time.time()
            totalTime = timeEnd - timeStart
            print(totalTime,'totalTime')
        elif self.frameNumber == 401: #for testing only to pause at a certain frame
            return (None,2)
        return (None,1) #return 1 to stay active





















class People():
    ## The constructor.
    def __init__(self):
        self.listOfPeople=list()
        self.lostListOfPeople=list()
        self.index=0

# Updates an item in the list of people/object or appends a new entry or assigns to a group or removes from a group      
    def update(self,img,fgmask,fX,fY,fW,fH,frameNumber,roiHist,height,bSROIs,homography,cameraPosition,listOfPeople):
        
        matches = []
        #lostFlag = 0
        occlCandidate = [] #list to hold a group of people that a person may be added to
        i = 0
        #print[fX,fY,fW,fH]
        box0 = Tools.bsRoiFinderBox([fX,fY,fW,fH],roiHist,bSROIs,fgmask,img,ROI_RESIZE_DIM)#corresponding roi for hog box
        box2 = [fX,fY,fX+fW,fY+fH]
        p1 = Tools.pixelToWorld((fX+(fW/2),fY+fH), homography)
        if len(box0) == 0:
            return
        else:
            if box0[0] <= 100 or box0[0]+box0[2] >= WIDTH-100 or box0[1] <= 2 or box0[1]+box0[3] >= HEIGHT-20 : #check if person1 is on edge of scene
                boxOnEdge = True
                
            else:
                boxOnEdge = False

        
        if len(matches) == 0: #new method 1
            i = 0
            for person in self.listOfPeople:
                box1 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
                lapping = Tools.overLap(box1,box2) #largest overlap
                if lapping > 0:
                    histDist = Tools.histogramComparison(roiHist,person.roiHist)
                    if len(matches)>0:
                        if lapping >= matches[0][0]:
                            if histDist < matches[0][3]:
                                matches = [(lapping, i,0,histDist)]  #flag of one means it was found in  lost people
                    else: #used in first iteration to set up overlap and histogram comparison
                        matches = [(lapping, i,0,histDist)]
                i = i + 1
        
        
        
        if len(matches) == 0 and boxOnEdge == False: #try to assign to person that is sharing a ROI
            i = 0
            for person in self.listOfPeople:
                if person.sharedROI == True:
                    p2 = Tools.pixelToWorld((person.fX+(person.fW/2),person.fY+person.fH), homography)
                    dist = Tools.objectDistance(p1,p2)
                    if dist < 5:
                        histDist = Tools.histogramComparison(roiHist,person.roiHist)
                    #print(histDist,'histDist 323')
                        if len(matches)>0: #used after first iteration to compare overlap and histogram 
                        #if lapping > matches[0][0]:
                            if histDist < matches[0][3]:
                                matches = [([], i,0,histDist)]  #flag of one means it was found in  lost people
                        else: #used in first iteration to set up overlap and histogram comparison
                            matches = [([], i,0,histDist)]
                i = i + 1
            if len(matches) > 0:
                
                person = self.listOfPeople[matches[0][1]]
                if person.V > 0:
                    person.roiCurrent = box0
                    person.lastGoodROI = box0
                    person.lastROICurrent = box0
                    person.fX,person.fY,person.fW,person.fH = person.roiCurrent[0],person.roiCurrent[1],person.roiCurrent[2],person.roiCurrent[3]
                    person.V=0
                    person.edgeCounter = 0
                    person.roiCounter = 0
                    #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 0',lostFlag,'lostFlag')
                return
                
        
        if len(matches) == 0: #check lost people for match
            i = 0
            for person in self.lostListOfPeople:
                histDist = Tools.histogramComparison(roiHist,person.roiHist)
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
                    pointA = Tools.pixelToWorld((person.location[-1][1],person.location[-1][2]),homography)
                    pointB = Tools.pixelToWorld((fX+(fW/2),fY+(fH/2)),homography)
                    distM = Tools.objectDistance(pointA,pointB)
                    if distM < 5:
                        pass
                        #lostFlag = 1
                    else:
                        matches = []
                        #lostFlag = 0

        
        if len(matches)>0 and len(occlCandidate) == 0: #1 match found and no occlusion update person attributes update person
            flag = matches[0][2]
            index = matches[0][1]
            if flag == 0: #get the person from matches
                person = self.listOfPeople[index]
            else:
                person = self.lostListOfPeople[index]
                self.insertPerson(person,self.listOfPeople)
                self.removePerson(person.ID,self.lostListOfPeople)
             
            if person.V > 0:#frameNumber > person.location[-1][0]: #if this is the first hog box for a person in this frame update person
                if len(person.roiHistList)< 5: #try to optimize the persons color histogram
                    
                    p1 = Tools.pixelToWorld((fX+(fW/2),fY+fH),homography)
                    distance = -1
                    for person2 in self.listOfPeople:
                        if person2.ID != person.ID:
                            p2 = Tools.pixelToWorld((person2.fX + (person2.fW/2),person2.fY+person2.fH),homography)
                            distance2 = Tools.objectDistance(p1,p2)
                            #distance.append(distance2)
                            #print('distance in hist update', distance2)
                            if distance != -1:
                                if distance2 < distance:
                                    distance = distance2
                                else:
                                    pass
                            else:
                                distance = distance2
                               # print('test')
                        
                   # print('distance in hist update', distance)
                    #print(min(distance),'min')
                    if distance == -1 or distance > 1:
                        roiHist2 = Tools.roiFGBGHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM)
                        person.roiHistList.append(roiHist2)
                        #Tools.displayHistogram(person.roiHist,frameNumber,person.ID)
                    if len(person.roiHistList) > 1: #optimize the histogram
                    
                        #print(person.roiHistList)                     
                        for hist in person.roiHistList:
                            for i in range(len(person.roiHist)):
                                person.roiHist[i] = person.roiHist[i]+hist[i]
                        for i in range(len(person.roiHist)):
                            person.roiHist[i] = person.roiHist[i]/len(person.roiHistList)
         
                    
                person.V=0
                person.edgeCounter = 0
                person.roiCounter = 0
                #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 1',lostFlag,'lostFlag')
                return
            else:
                
                return


        
        
        elif len(matches) == 0 and len(occlCandidate) == 0:#4 no match found so create person after refining the hog box
            roiHist = Tools.foreGroundHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM)
            tmp_node=Person(self.index,fX,fY,fW,fH,0,roiHist,height) #step 3 only update persons histogram on creation, not in subsequent updates.
            #Tools.dbScanAlgorithmRoiCompare(bSROIs,[tmp_node],fgmask,img,ROI_RESIZE_DIM)
            tmp_node.roiCurrent = Tools.bsRoiFinderPerson(tmp_node,bSROIs,fgmask,img,ROI_RESIZE_DIM)
            if len(tmp_node.roiCurrent) > 0:
                tmp_node.fX,tmp_node.fY,tmp_node.fW,tmp_node.fH = tmp_node.roiCurrent[0],tmp_node.roiCurrent[1],tmp_node.roiCurrent[2],tmp_node.roiCurrent[3]
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                return
            else:
                tmp_node.roiCurrent = [fX,fY,fW,fH]
                #tmp_node.roiCurrent = []
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                #print('new person added roiCurrent cheated, index is '+ str(self.index) +' case 4e')
                return
                    
    def refresh(self,img,imgCopy,fgmask,frameNumber,bSROIs,homography,cameraPosition,term_crit,waitingList): #updates people's boxes and checks for occlusion
        personList = list(self.listOfPeople) #make copy of people list to use for while loop
        
        while len(personList) > 0:
            
            person1 = self.getPerson(personList[0].ID,self.listOfPeople)
            #print(person1.ID,'moving = ', person1.moving)
            flag = 0
            person1.V=person1.V+1
            if len(person1.roiCurrent) != 0:
                person1Box2 = [person1.roiCurrent[0], person1.roiCurrent[1], person1.roiCurrent[2], person1.roiCurrent[3]]
            else:
                person1Box2 = [person1.lastROICurrent[0], person1.lastROICurrent[1], person1.lastROICurrent[2], person1.lastROICurrent[3]]
            for person2 in self.listOfPeople: #determine whether person shares BSroi with other person and add to new group if so
                if person1.ID != person2.ID:
                    if len(person2.roiCurrent) != 0:
                        person2Box = [person2.roiCurrent[0], person2.roiCurrent[1], person2.roiCurrent[2], person2.roiCurrent[3]]
                    else:
                        person2Box = [person2.lastROICurrent[0], person2.lastROICurrent[1], person2.lastROICurrent[2], person2.lastROICurrent[3]]
                        
                    if person1Box2 == person2Box:
                        flag = 1
                              
                                
            #print(person1.ID, flag, 'flag for current person')                

            if flag == 0 and len(person1.roiCurrent) > 0 and person1.V == 1:
                
                bottomPoint = Tools.pixelToWorld((person1.fX+(person1.fW/2),person1.fY+person1.fH), homography) #find location of bottom center of the roi
                topPoint = Tools.pixelToWorld((person1.fX+(person1.fW/2),person1.fY), homography)         #find location of the top center of the roi
                tmpHeight = Tools.objectHeight(bottomPoint, topPoint, cameraPosition)   #calculate the height of the roi
                #print(box,'BSFinderbox 114')
                #print(person.location)
                if tmpHeight >= 1.25: #filter out bad regions based on height
                    
                    #find avg height for person
                    if len(person1.heightList) < 35:
                        #if person.V == 1:
                        person1.heightList.append(tmpHeight)
                        
                    else:
                        if person1.height == -1:
                            person1.height = np.average(person1.heightList)
                            
                    leftPoint = Tools.pixelToWorld((person1.fX,person1.fY+person1.fH), homography)
                    rightPoint = Tools.pixelToWorld((person1.fX+person1.fW,person1.fY+person1.fH), homography)
                    tmpWidth = Tools.objectDistance(leftPoint,rightPoint)        
                    if len(person1.widthList) < 35:
                        person1.widthList.append(tmpWidth)
                        
                    else:
                        if person1.width == -1:
                            person1.width = np.average(person1.widthList)    
            
      
            
            if len(person1.roiCurrent) == 0 and person1.nearEdge == True and person1.edgeCounter > 15:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
                if person1 in waitingList:
                    waitingList.remove(person1)
                self.insertPerson(person1,self.lostListOfPeople)
                #print(person1.ID,'sent to lost people left edge of scene')
                self.removePerson(person1.ID,self.listOfPeople)
                personList.remove(person1)
                continue #skip to next person            

            if len(person1.roiCurrent) == 0  and person1.roiCounter > 1000 and person1.V > 120:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
                if person1 in waitingList:
                    waitingList.remove(person1)
                self.insertPerson(person1,self.lostListOfPeople)
                #print(person1.ID,'sent to lost people lost in scene')
                self.removePerson(person1.ID,self.listOfPeople)
                personList.remove(person1)
                continue #skip to next person
            
            if len(person1.roiCurrent) > 0 and flag == 1: #ROI is shared
                person1.sharedROI = True
                if person1.moving == True:
                    person1.lastGoodROI = person1.roiCurrent
                    #print('case1a in refresh, meanshift on last location, roi is shared')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6)
   
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < person1.roiCurrent[0]: # code to keep box from wondering
                        person1.fX = person1.roiCurrent[0]
                        #print('fixed box location 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > person1.roiCurrent[0]+person1.roiCurrent[2]:
                        diff = (person1.fX+person1.fW) - (person1.roiCurrent[0]+person1.roiCurrent[2])
                        a = Tools.np.array((person1.fX+person1.fW))
                        b = Tools.np.array((person1.roiCurrent[0]+person1.roiCurrent[2]))
                        dist = Tools.np.linalg.norm(a-b)
                        
                        print(diff)
                        print(dist)
                        if person1.fX > person1.roiCurrent[0]+diff:
                            person1.fX = person1.fX - diff
                            
                            #print('fixed box location 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        elif person1.roiCurrent[0] == 0: 
                            person1.fX = person1.roiCurrent[0]
                    
                    if person1.fY < person1.roiCurrent[1]: # code to keep box from wondering
                        person1.fY = person1.roiCurrent[1]
                        #print('fixed box location 3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > person1.roiCurrent[1]+person1.roiCurrent[3]:
                        diff = (person1.fY+person1.fH) - (person1.roiCurrent[1]+person1.roiCurrent[3])
                        a = Tools.np.array((person1.fY+person1.fH))
                        b = Tools.np.array((person1.roiCurrent[1]+person1.roiCurrent[3]))
                        dist = Tools.np.linalg.norm(a-b)
                        
                        if person1.fY > person1.roiCurrent[1]+diff:
                            #person1.fY = person1.roiCurrent[1]
                            person1.fY = person1.fY - diff
                            #print('fixed box location 4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        
                elif person1.moving == False:
                    #print('case1b in refresh, meanshift on last location, roi is shared')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6)
   
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < person1.roiCurrent[0]: # code to keep box from wondering
                        person1.fX = person1.roiCurrent[0]
                        #print('fixed box location 1b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > person1.roiCurrent[0]+person1.roiCurrent[2]:
                        diff = (person1.fX+person1.fW) - (person1.roiCurrent[0]+person1.roiCurrent[2])
                        if person1.fX > person1.roiCurrent[0]+diff:
                            person1.fX = person1.fX - diff
                            #print('fixed box location 2b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < person1.roiCurrent[1]: # code to keep box from wondering
                        person1.fY = person1.roiCurrent[1]
                        #print('fixed box location 3b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > person1.roiCurrent[1]+person1.roiCurrent[3]:
                        diff = (person1.fY+person1.fH) - (person1.roiCurrent[1]+person1.roiCurrent[3])
                        if person1.fY > person1.roiCurrent[1]+diff:
                            person1.fY = person1.fY - diff
                            #print('fixed box location 4b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                
            
            elif len(person1.roiCurrent) == 0 and flag == 0:# : # do for every person with no BS roi
                
                person1.roiCounter += 1
                if person1.edgeCounter == 0 and person1.moving == True:
                    #print('case2a in refresh, no current ROI, adjust box and meanshift')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                    previousFX = person1.lastGoodROI[0]
                    previousFY = person1.lastGoodROI[1]
                    previousFW = person1.lastGoodROI[2]
                    previousFH = person1.lastGoodROI[3]
                    #person1.fX,person1.fY,person1.fW,person1.fH = person1.lastGoodROI[0],person1.lastGoodROI[1],person1.lastGoodROI[2],person1.lastGoodROI[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < previousFX: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 5 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > previousFX+previousFW :
                        diff = (person1.fX+person1.fW) - (previousFX+previousFW)
                        if person1.fX > previousFX + diff:
                            #person1.fX = person1.roiCurrent[0]
                            person1.fX = person1.fX - diff
                            #person1.fX = int(previousFX - .2*previousFW)
                            #print('fixed box location 6 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < previousFY: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 7 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > previousFY+previousFH:
                        diff = (person1.fY+person1.fH) - (previousFY+previousFH)
                        if person1.fY > previousFY + diff:
                            person1.fY = person1.fY - diff
                        #person1.fY = int(previousFY - .1*previousFH)
                            #print('fixed box location 8 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                
                elif person1.edgeCounter == 0 and person1.moving == False:
                    #print('case2b in refresh, no current ROI, adjust box and meanshift')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                    previousFX = person1.lastROICurrent[0]
                    previousFY = person1.lastROICurrent[1]
                    previousFW = person1.lastROICurrent[2]
                    previousFH = person1.lastROICurrent[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                   
                    if person1.fX < previousFX: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 9 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > previousFX+previousFW :
                        diff = (person1.fX+person1.fW) - (previousFX+previousFW)
                        if person1.fX > previousFX+diff:
                            person1.fX = person1.fX - diff
                            #print('fixed box location 10 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < previousFY: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 11 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > previousFY+previousFH:
                        diff = (person1.fY+person1.fH) - (previousFY+previousFH)
                        if person1.fY > previousFY+diff:
                            person1.fY = person1.fY - diff
                            #print('fixed box location 12 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                
                    
                elif person1.edgeCounter > 0:
                    previousFX = person1.lastROICurrent[0]
                    previousFY = person1.lastROICurrent[1]
                    previousFW = person1.lastROICurrent[2]
                    previousFH = person1.lastROICurrent[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                   
                    if person1.fX < previousFX or person1.fX+person1.fW > previousFX+previousFW: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 13 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                   
                    if person1.fY < previousFY or person1.fY+person1.fH > previousFY+previousFH: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 14 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
            elif len(person1.roiCurrent) == 0 and flag == 1:# : # do for every person with no BS roi and shares previous roicurrent
                person1.roiCounter += 1
                #print('case2a in refresh, no current ROI, adjust box and meanshift')
                cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                previousFX = person1.lastROICurrent[0]
                previousFY = person1.lastROICurrent[1]
                previousFW = person1.lastROICurrent[2]
                previousFH = person1.lastROICurrent[3]
                Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                                
                if person1.fX < previousFX: # code to keep box from wondering
                    person1.fX = previousFX
                    #print('fixed box location 15 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                elif person1.fX+person1.fW > previousFX+previousFW :
                    person1.fX = previousFX
                    #print('fixed box location 16 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                
                if person1.fY < previousFY: # code to keep box from wondering
                    person1.fY = previousFY
                    #print('fixed box location 17 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                elif person1.fY+person1.fH > previousFY+previousFH:
                    person1.fY = previousFY
                    #print('fixed box location 18 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')         
                    
            else: # person.roicurrent != [] and not shared
                #print('case3 in refresh, person has current ROI, personbox = person.roi',person1.ID)
                person1.sharedROI = False
                person1.roiCounter = 0
                box = [person1.roiCurrent[0], person1.roiCurrent[1], person1.roiCurrent[2], person1.roiCurrent[3]]
                person1.lastGoodROI = person1.roiCurrent
                person1.fX=box[0]
                person1.fY=box[1]
                person1.fW=box[2]
                person1.fH=box[3]
                
            person1.location.append([frameNumber,(person1.fX+(person1.fX+person1.fW))/2,(person1.fY+(person1.fY+person1.fH))/2])            
            if frameNumber % 1 == 0:     
                if len(person1.kalmanLocation) > 0:
                    world = Tools.pixelToWorld((person1.kalmanLocation[-1][0],person1.kalmanLocation[-1][1]),homography) # for movement classification
                    worldArray = np.array([world[0],world[1]],ndmin = 2)
                    worldArray.shape = (1,2)
                    person1.locationArray = np.append(person1.locationArray,worldArray, axis = 0) # for movement classification
                    person1.worldLocation = worldArray

            if person1.fX <= 25 or person1.fX+person1.fW >= WIDTH-1 or person1.fY+ person1.fH <= 50 or person1.fY+person1.fH >= HEIGHT-1 : #check if person1 is on edge of scene
                person1.nearEdge = True
                person1.edgeCounter +=1
            else:
                person1.nearEdge = False
            
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
    
    def __init__(self,ID,fX,fY,fW,fH,visible,roiHist,height):
        self.ID=ID
        self.fX=fX
        self.fY=fY
        self.fW=fW
        self.fH=fH
        self.V=visible
        self.location=[] 
        self.kalmanLocation = []
        self.height = -1
        self.heightList = []
        self.width = -1
        self.widthList = []
        self.direction = []
        self.roiHist = roiHist
        self.roiHistList = [roiHist]
        self.kalmanX = Tools.KalmanFilter(fX+(fW/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0) 
        self.kalmanY = Tools.KalmanFilter(fY+(fH/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0) 
        self.roiCurrent = []
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
                    
