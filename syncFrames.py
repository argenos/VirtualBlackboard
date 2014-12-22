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


def getf(cam,num,r1,r2,frames):
    print "Start Thread %d"%num
    for i in xrange(frames):
        r1.set()
        r2.wait()
        print "Ready %d"%num
        print "Done waiting %d"%num
        r1.clear()
        print "Getting image %d"%num
        r,camera_capture = cam.read()
        file = "images/c%d_image%d.png"%(num,i)
        cv2.imwrite(file, camera_capture)
        print "Done %d"%num
    r1.set()
    del(cam)
    print "Exit Thread %d"%num


def main():  
    cam1 = cv2.VideoCapture(0) 
    cam2 = cv2.VideoCapture(1) 
    
    c1 = threading.Event()
    c2 = threading.Event()
    
    frames = 50
    
    c1thread = threading.Thread(target=getf,args=(cam1,1,c1,c2,frames))    
    c2thread = threading.Thread(target=getf,args=(cam2,2,c2,c1,frames))
    
    c1thread.start()
    c2thread.start()
    c1thread.join()
    c2thread.join()

    print "Exit main..."
    
    

if __name__ == '__main__':
    main()
