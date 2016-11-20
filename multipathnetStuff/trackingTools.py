import random
import pickle
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

class MaxBipartite(object):

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def bpm(self, bpGraph, u, seen, matchR):
        bpGraph
        for v in range(self.N):
            #print([u,v], min(bpGraph[u]),min([x[v] for x in bpGraph]))
            
            if (bpGraph[u][v] != -1 and (bpGraph[u][v] >= max(bpGraph[u]) and (bpGraph[u][v] >= max([x[v] for x in bpGraph])))  and not (seen[v]) ): # greater and max for correlation, use less and min than for bhatthacharyyas method
                #print("potential good match is ",[u,v], bpGraph[u][v])
            #if bpGraph[u][v] and not seen[v]:
                seen[v] = True

                if (matchR[v] < 0 or self.bpm(bpGraph, matchR[v], seen, matchR)):
                    matchR[v] = u
                    return True

        return False

    def maxBPM(self, bpGraph):
        matchR = [0] * self.N

        for i in range(self.N):
            matchR[i] = -1

        result = 0
        for u in range(self.M):

            seen = [False] * self.N
            for i in range(self.N):
                seen[i] = False

            if self.bpm(bpGraph, u, seen, matchR):
                result += 1
        #for i in range(len(matchR)):
            #print(matchR[i])

        #return result
        return matchR

def rand_array(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]


def main():
    """bpGraph = [[0.063, 0.976, 0.630, 0.798, 0.926, 0.447],
               [0.906, 0.017, 0.585, 0.122, 0.796, 0.398],
               [0.401, 0.239, 0.033, 0.629, 0.727, 0.354],
               [0.147, 0.254, 0.272, 0.067, 0.830, 0.679], 
               [0.063, 0.756, 0.134, 0.515, 0.053, 0.924], 
               [0.300, 0.844, 0.578, 0.825, 0.183, 0.028]]"""
    """bpGraph = [[0.3638903659471453, 0.9761139171410832, 0.3303319231563049],
               [0.9065725264825174, 0.1735533682269319, 0.3853397311047968],
               [0.40188120553635587, 0.23944120955902193, 0.9439235781363713],
               [0.1476368120205701, 0.25475687161080596, 0.2720203489219124], 
               [0.08528319155757547, 0.7564186590012879, 0.13476264281037809], 
               [0.3008867406705923, 0.8444700107777172, 0.5783045229376217]]"""

    """bpGraph = [[False, True, True, False, False, False],
    [True, False, False, True, False, False],
    [False, False, True, False, False, False],
    [False, False, True, True, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, True]]

    m = MaxBipartite(6, 6)

    a = rand_array(6)
    print(a)
    print(m.maxBPM(a))
    [False, False, False, False, False, True]]"""
    """bpGraph = [[True, False, False, False, False, False],
    [False, True, False, True, False, False],
    [False, True, False, False, False, False],
    [True, False, False, False, False, False],
    [True, False, True, False, False, False],
    [False, False, False, False, True, False]]"""
    bpGraph = [[0.063],
               [0.976],
               [0.995]]
    
    print([x[0] for x in bpGraph])
    
    print(len(bpGraph),len(bpGraph[0]), "array shape")
    #m = MaxBipartite(6, 6)
    m = MaxBipartite(len(bpGraph),len(bpGraph[0]))
    print("Num of matches found", m.maxBPM(bpGraph))

#main()
class ImgToImgHomography():
    def __init__(self,image1,image2,filename):
        self.homography = []
        self.imagePoints1 = []
        self.imagePoints2 = []
        self.image1 = image1
        self.image2 = image2
        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.dXRGBToIR = None
        self.dYRGBToIR = None
        self.dXIRToRGB = None
        self.dYIRToRGB = None
        self.filename = "BGRtoIR_Homography_"+filename
        self.defaultImage1 = image1.copy()
        self.defaultImage2 = image2.copy()

    def displayOptions(self):
        print("=============================================================================================")
        print("Select a point on one image and then the corresponding point in the other image ")
        print("Click left mouse to select point on first image" )
        print("Click right mouse to select point on second image" )
        print(" d = remove a point from the first list")
        print(" f = remove a point from the second list")
        print(" h = compute homography, p = find corresponding image point,")
        print(" l = load from previous saved homography, s = save images")
        print("z = save point to list1, x = save point to list2, i = offset demo RGB to IR")
        print("u = offset demo IR to RGB, o = calcOffset, c = clear screen, k = default lists, q = quit")
    #code for calculating homography matrix
    
    def findHom(self): # need to verify that the matrix is correct
        """Calculates the homography if there are 4+ point pairs"""
        src = np.array(self.imagePoints1, dtype=np.float64)      # using cv2.getPerspectiveTransform returns the same matrix as the code above but only allows for 4 points.
        dest = np.array(self.imagePoints2, dtype=np.float64)
        H3, mask = cv2.findHomography(src, dest, cv2.RANSAC,5.0)
        self.homography = H3
        origin = ([[self.imagePoints1[0][0]],[self.imagePoints1[0][1]],[1]])
        originMatrix = np.array(origin,dtype=np.float32)
        originTest = np.dot(H3, originMatrix)
        originTestXY = np.divide(originTest,originTest[2])
        f = open( self.filename+".p", "wb" )
        pickle.dump(H3, f )
        f.close()
        f = open( self.filename+"imgPnts1.p", "wb" )
        pickle.dump(self.imagePoints1, f )
        f.close()
        f = open( self.filename+"imgPnts2.p", "wb" )
        pickle.dump(self.imagePoints2, f )
        f.close()
        print("Matrix H3 derived from cv2.findHomography(src, dest, cv2.RANSAC,5.0).")
        print(H3)
        print("Image coordinates of world origin.")
        print(originMatrix)
        print("Origin test result world coordinates.")
        print(originTestXY)
    def calcOffSet(self):
        self.dXRGBToIR = self.x1-self.x2
        self.dYRGBToIR = self.y1-self.y2
        self.dXIRToRGB = self.x2-self.x1
        self.dYIRToRGB = self.y2-self.y1
        print(self.dXRGBToIR,self.dYRGBToIR)
        print(self.dXIRToRGB,self.dYIRToRGB)
        
    def drawRoi1(self,event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN: #left button for selecting points on image 1
            self.x1=x
            self.y1=y            
            cv2.circle(self.image1,(self.x1,self.y1),2,(0,0,255), 2) 
            
            tmp = (self.x1,self.y1)
            #cv2.putText(self.image1, str(tmp),(self.x1+10,self.y1), 0, .5, (0,0,255), 1,8, False)
                
            
    
    def drawRoi2(self,event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN: #left button for selecting points on image 2
            self.x2=x
            self.y2=y            
            cv2.circle(self.image2,(self.x2,self.y2),2,(0,0,255), 2) 
            tmp = (self.x2,self.y2)
            #cv2.putText(self.image2, str(tmp),(self.x2+10,self.y2), 0, .5, (0,0,255), 1,8, False) 
            
                                  
    def pointSelector(self): #Helper
        self.displayOptions()
        
        try:
            f = open( self.filename+".p", "rb")
            self.homography = pickle.load( f ) #code for loading homography from file.
            f.close()
            print("homograpy successfully loaded")
        except:
            print("no previously saved homography exists, you must create the homography")
        
        cv2.namedWindow('image1')#, flags = cv2.WINDOW_NORMAL)# toggle flag when using on a limited resolution display. not for exact measurements though. 
        cv2.namedWindow('image2')
                
        
        cv2.setMouseCallback('image1',self.drawRoi1) # listens for mouse events
        cv2.setMouseCallback('image2',self.drawRoi2) # listens for mouse events
        
        
        while(True):
            
            cv2.imshow('image1',self.image1)
            cv2.imshow('image2',self.image2)         
            
            k = cv2.waitKey(10) & 0xFF
            
            if k == ord('p') and len(self.homography) > 0:
                cv2.circle(self.image1,(self.x1,self.y1),2,(255,0,0), 2) 
                otherPoint = self.pointToPoint()
                x = int(otherPoint[0])
                y = int(otherPoint[1])
                tmp = (x,y)
                cv2.putText(self.image2, str(tmp),(otherPoint[0]+10,otherPoint[1]), 0, .5, (255,0,0), 1,8, False)
                cv2.circle(self.image2,(otherPoint[0],otherPoint[1]),2,(255,0,0), 2) 
                
                print(otherPoint)
                
            if k == ord('i') and self.dXRGBToIR is not None:
                #cv2.circle(self.image1,(self.x1,self.y1),2,(255,0,0), 2) 
                firstPoint = (self.x1,self.y1)
                otherPoint = (int(self.x1-self.dXRGBToIR),int(self.y1-self.dYRGBToIR))
                #x = int(otherPoint[0])
                #y = int(otherPoint[1])
                #tmp = (x,y)
                #cv2.putText(self.image1, str(firstPoint),(firstPoint[0]+10,firstPoint[1]), 0, .5, (255,0,0), 1,8, False)
                #cv2.putText(self.image2, str(otherPoint),(otherPoint[0]+10,otherPoint[1]), 0, .5, (255,0,0), 1,8, False)
                cv2.circle(self.image2,(otherPoint[0],otherPoint[1]),2,(255,0,0), 2) 
                
                print(otherPoint)
            if k == ord('u') and self.dXIRToRGB is not None:
                #cv2.circle(self.image1,(self.x1,self.y1),2,(255,0,0), 2) 
                firstPoint = (self.x2,self.y2)
                otherPoint = (int(self.x2-self.dXIRToRGB),int(self.y2-self.dYIRToRGB))
                #x = int(otherPoint[0])
                #y = int(otherPoint[1])
                #tmp = (x,y)
                #cv2.putText(self.image1, str(firstPoint),(firstPoint[0]+10,firstPoint[1]), 0, .5, (255,0,0), 1,8, False)
                #cv2.putText(self.image2, str(otherPoint),(otherPoint[0]+10,otherPoint[1]), 0, .5, (255,0,0), 1,8, False)
                cv2.circle(self.image1,(otherPoint[0],otherPoint[1]),2,(255,0,0), 2) 
                
                print(otherPoint)
            
            if k == ord('z') and self.x1 != -1 and self.y1 != -1:
                self.imagePoints1.append([self.x1,self.y1])
                print("imagePoints1 = ",self.imagePoints1)
                print("imagePoints2 = ",self.imagePoints2)
                
            if k == ord('x') and self.x2 != -1 and self.y2 != -1:
                self.imagePoints2.append([self.x2,self.y2])
                print("imagePoints1 = ",self.imagePoints1)
                print("imagePoints2 = ",self.imagePoints2)
                
            if k== ord('l') : #and init and x != "None":  attempt to load prior homography
                try:
                    f = open( self.filename+".p", "rb" )
                    self.homography = pickle.load( f ) #code for loading homography from file.
                    f.close()
                    print('homography.p successfully loaded')
                except:
                    print('homography not available')
                                                                            
            if k== ord('h'):
               
                if len(self.imagePoints1) >= 4 and len(self.imagePoints1) == len(self.imagePoints2):                    
                    print("Computing homography matrix for image to image conversion")
                    self.findHom()
                    
                else:
                    print("not enough image points to calculate homography or list are not same length")
            
            if k== ord('o'):
               
                if self.x1 != -1 and self.x2 != -1 :                    
                    print("Computing offset for image to image conversion")
                    self.calcOffSet()
                    
                else:
                    print("select a point in each image first")
            if k== ord('d') and len(self.imagePoints1) > 0:
                print(self.imagePoints1.pop(-1))
                print(self.imagePoints1)
                print("Removed last chosen pixel homography point from image 1 (rgb).")
            
            if k== ord('f') and len(self.imagePoints2) > 0:
                print(self.imagePoints2.pop(-1))
                print(self.imagePoints2)
                print("Removed last chosen pixel homography point from image 2 (IR).")
            
# Save Image for Homography 
            if k== ord('s'):
                print("Homography Image")                
                saveName=self.filename+"RGB.jpeg"  
                cv2.imwrite(saveName, self.image1)
                saveName=self.filename+"IR.jpeg"  
                cv2.imwrite(saveName, self.image2)                    
  
# clears screen and restores all points lists to default
            if k == ord('c'):
                
                self.image1 = self.defaultImage1.copy()
                self.image2 = self.defaultImage2.copy()
                self.displayOptions()
# clears screen and restores all points lists to default                
            if k == ord('k'):
                self.x1 = -1
                self.y1 = -1
                self.x2 = -1
                self.y2 = -1
                self.dXIRToRGB = None
                self.dXRGBToIR = None
                self.dYIRToRGB = None
                self.dYRGBToIR = None
                self.imagePoints1 = []
                self.imagePoints2 = []
                self.image1 = self.defaultImage1.copy()
                self.image2 = self.defaultImage2.copy()
                self.displayOptions()
                
# q for quit            
            if k== ord('q'):
                
                break

        cv2.destroyWindow("image1")
        cv2.destroyWindow("image2")
        return self.homography

    # code for calculating world coord from pixel coord
    def pointToPoint(self):
        if not len(self.homography) == 0:
            point = ([[self.x1],[self.y1],[1]])
            pointMatrix = np.array(point,dtype=np.float32)
            otherPoint = np.dot(self.homography, pointMatrix) #dot or multiply?
            other = np.divide(otherPoint,otherPoint[2])
            otherX, otherY = other[0],other[1]
            #worldVal = str(world[0])+","+str(world[1])
            #print(worldX,worldY)
            return (otherX, otherY)
        else:
            print("You must first create homography matrix.")

def nonMaxSup(rects,thresh = 0.5):
    #print(rects)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=thresh) 
    i = 0
    for (xA, yA, xB, yB) in pick:
        pick[i][2] = xB-xA
        pick[i][3] = yB-yA
        i +=1
    return pick