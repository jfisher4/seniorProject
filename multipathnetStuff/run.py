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
goodVideoList = ["00004.MTS","00005.MTS","00006.MTS","00007.MTS","00008.MTS","01072016A5_J1C.mp4","00010.MTS","00011.MTS","00012.MTS","00013.MTS","00014.MTS","00015.MTS","00016.MTS","00017.MTS","00018.MTS","00019.MTS","00020.MTS",]
#done before crash
#for i in range(len(goodVideoList)):
#    run("/home/robotics_group/multipathnet/deepmask/data/",goodVideoList[i])
run("/home/robotics_group/multipathnet/deepmask/data/","00012.MTS")
