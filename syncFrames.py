# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:16:01 2014

@author: Argen
"""
import threading
import cv2
import time
import numpy as np

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


def getVideo(cam,name,showImg=True):
    ret,frame = cam.read()
    while(True):
        ret, frame = cam.read()
        newFrame = cv2.resize(frame,(640,480))

        #if showImg:
        cv2.imshow('Video',newFrame)

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
    back1 = getVideo(cam1,'Background 1')
    cv2.imwrite('images/background/background1.png', back1)
    print "Please step away and click 's' when it's ready..."
    back2 = getVideo(cam2, 'Background 2')
    cv2.imwrite('images/background/background2.png', back2)

    cam1.release()
    cam2.release()

def getBlackboardCorners():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    corners = ['top_left','top_right', 'bottom_left', 'bottom_right']

    for i in xrange(4):
        print "Please touch the "+corners[i]+' corner of the blackboard with a marker.'

        corner1 = getVideo(cam1, 'Corner1')
        cv2.imwrite('images/corners/c1_corner%.2d.png'%i, corner1)
        corner2 = getVideo(cam2, 'Corner2')
        cv2.imwrite('images/corners/c2_corner%.2d.png'%i, corner2)


    cam1.release()
    cam2.release()

def calibrate(images):
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    print "Calibrating"
    for i in xrange(images):
        print "Waiting for key"
        img1 = getVideo(cam1,'Calibration1')
        img2 = getVideo(cam2,'Calibration2')
        two = np.hstack((img1,img2))
        two = cv2.resize(two,(1280,460))
        print "Showing two images"
        cv2.imshow("Captured frame",two)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('images/calibration_stereo/c1_calib%.2d.jpg'%i, img1)
        cv2.imwrite('images/calibration_stereo/c2_calib%.2d.jpg'%i, img2)

    cam1.release()
    cam2.release()


def init(frames,f = True,calib=False,*args):
    if calib:
        calibrate(25)

    if f:
        print "Getting background."
        getBackground()
        print "Getting blackboard corners."
        getBlackboardCorners()
        print "Getting frames."
        for arg in args:
            path = 'images/'+arg+'_images/'
            getSyncedFrames(frames)



def changeExtension():
    for i in xrange(25):
        img1 = cv2.imread('images/calibration/c1_calib%.2d.png'%i)
        cv2.imwrite('images/calibration/c1_calib%.2d.jpg'%i, img1)
        img2 = cv2.imread('images/calibration/c2_calib%.2d.png'%i)
        cv2.imwrite('images/calibration/c2_calib%.2d.jpg'%i, img2)



if __name__ == '__main__':
    #init(100)
    calibrate(40)
    #changeExtension()
    print "All done."
