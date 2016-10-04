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

run("/home/robotics_group/multipathnet/deepmask/data/","000026_push.avi")
