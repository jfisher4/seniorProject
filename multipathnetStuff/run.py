from HumanTracker import *
import cv2

def run(directory,videoname):
    detector = HumanTracker(directory,videoname)
    active = 1 # 1 for active , 0 for inactive, 2 for paused
#    active = True
#    paused = False
    while(1): 
        if active == 1:
            #peopleA, active, paused = simulationA.retrieve(paused)
            peopleA, active = detector.readAndTrack()
        elif active == 0:
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            print("Continuing...")
            active = 1
        elif k == ord('q'):
            print("Exiting Program...")
            
            cv2.destroyAllWindows()
            break
#goodVideoList = ["01072016A5_J1C.mp4","00028.MTS","00029.MTS","00030.MTS","00031.MTS","00032.MTS","00033.MTS","00034.MTS","00035.MTS","00036.MTS","00037.MTS","00038.MTS","00039.MTS","00040.MTS","00041.MTS","00042.MTS","00043.MTS","00044.MTS","00045.MTS"]

#for i in range(len(goodVideoList)):
    #run("/home/robotics_group/multipathnet/deepmask/data/",goodVideoList[i])
#run("C:\\Users\\gifis\\Documents\\seniorProject\\multipathnetStuff\\","00028.MTS") # for ryan's windows laptop
run("/home/robotics_group/multipathnet/deepmask/data/","00004.MTS")
