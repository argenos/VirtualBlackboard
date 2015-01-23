# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 10:02:05 2015

@author: Argen
"""
import glob
import cv2
import numpy as np
import Color as color

path = 'images/frames/blue/'
length = glob.glob(path+'*.png').__len__()/2

def playSingleVideo(path, mask=True):
    #print length
    i = 0
    cv2.namedWindow("Frame sequence")
    print 'Press any key to play...'
    cv2.waitKey(0)

    while True:
        im = cv2.imread(path+'c1_image%.3d.png'%i)
        
        if mask:
            img = cv2.resize(im,(640,480))
            img = color.getColorMask(img,'b')
        cv2.imshow("Frame sequence",img)
        i = i + 1
        if i >= length:
            i = 0
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def playTwoVideos(path, mask=False, marker=True):
    i = 0
    fourcc = cv2.cv.FOURCC('m','p','4','v')
    out = cv2.VideoWriter('video/result.mp4',fourcc)
    cv2.namedWindow("Frame sequence")
    print 'Press any key to play...'
    cv2.waitKey(0)
    while True:

        im1 = cv2.imread(path+'c1_image%.3d.png'%i)
        im2 = cv2.imread(path+'c2_image%.3d.png'%i)

        if mask:
            img1 = cv2.resize(im1,(640,480))
            img2 = cv2.resize(im2,(640,480))
            img1 = color.getColorMask(img1,'b')
            img2 = color.getColorMask(img2,'b')
        else:
            img1 = cv2.resize(im1,(640,480))
            img2 = cv2.resize(im2,(640,480))

        img = np.hstack((img1,img2))
        out.write(img)
        cv2.imshow("Frame sequence",img)
        i = i + 1
        if i >= length:
            i = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #playSingleVideo(path)
    playTwoVideos(path)