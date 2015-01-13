# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:16:01 2014

@author: Argen
"""
import threading
import cv2
import time

camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(0)
path = 'images/test/'


def getf(cam,num,r1,r2,frames):
    print "Start Thread %d"%num
    for i in xrange(frames):
        r1.set()
        r2.wait()
        #print "Ready %d"%num
        #print "Done waiting %d"%num
        r1.clear()
        #print "Getting image %d"%num
        r,camera_capture = cam.read()
        file = path+"c%d_image%.3d.png"%(num,i)
        cv2.imwrite(file, camera_capture)
        #print "Done %d"%num
    r1.set()
    del(cam)
    print "Exit Thread %d"%num


def getSyncedFrames(frames):
    cam1 = cv2.VideoCapture(0) 
    cam2 = cv2.VideoCapture(1) 
    
    c1 = threading.Event()
    c2 = threading.Event()
    
    #frames = 50
    
    c1thread = threading.Thread(target=getf,args=(cam1,1,c1,c2,frames))    
    c2thread = threading.Thread(target=getf,args=(cam2,2,c2,c1,frames))
    
    c1thread.start()
    c2thread.start()
    c1thread.join()
    c2thread.join()

    print "Done. Total frames: %d"%frames


def getVideo(cam):
    ret,frame = cam.read()
    while(True):
        ret, frame = cam.read()
        newFrame = cv2.resize(frame,(640,480))

        cv2.imshow('Background',newFrame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # When everything done, release the capture
    #cam.release()
    cv2.destroyAllWindows()
    return frame


def getBackground():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    print "Please step away and click 's' when it's ready..."
    back1 = getVideo(cam1)
    cv2.imwrite('images/background1.png', back1)
    print "Please step away and click 's' when it's ready..."
    back2 = getVideo(cam2)
    cv2.imwrite('images/background2.png', back2)

def getBlackboardCorners():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    print "Please touch the left top corner of the blackboard with a marker."
    corner1 = getVideo(cam1)
    cv2.imwrite('images/corner1.png', corner1)
    print "Please touch the right bottom corner of the blackboard with a marker."
    corner2 = getVideo(cam2)
    cv2.imwrite('images/corner2.png', corner2)


def calibrate(images):
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    for i in xrange(images):
        img1 = getVideo(cam1)
        cv2.imwrite('images/calibration/c1_calib%.2d.png'%i, img1)
        img2 = getVideo(cam2)
        cv2.imwrite('images/calibration/c2_calib%.2d.png'%i, img2)

    cam1.release()
    cam2.release()


def init(frames,f = True,calib=False,*args):
    if f:
        print "Getting background."
        getBackground()
        print "Getting blackboard corners."
        getBlackboardCorners()
        print "Getting frames."
        for arg in args:
            path = 'images/'+arg+'_images/'
            getSyncedFrames(frames)


    if calib:
        calibrate(25)



if __name__ == '__main__':
    init(5,False,True)
