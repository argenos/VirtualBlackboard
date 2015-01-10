# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 10:02:05 2015

@author: Argen
"""
import glob
import cv2
import numpy as np
import Color as color

path = 'images/images_azul/'
length = glob.glob(path+'*.png').__len__()/2

def playSingleVideo(path, mask=True):
    #print length
    i = 0
    while True:
        im = cv2.imread(path+'c1_image%.2d.png'%i)
        img = cv2.resize(im,(640,480))
        if mask:
            img = color.getColorMask(img,'b')
        cv2.imshow("Picture",img)
        i = i + 1
        if i >= length:
            i = 0
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def playTwoVideos(path, mask=True):
    i = 0
    while True:
        im1 = cv2.imread(path+'c1_image%.2d.png'%i)
        img1 = cv2.resize(im1,(640,480))
        
        im2 = cv2.imread(path+'c2_image%.2d.png'%i)
        img2 = cv2.resize(im2,(640,480))
        
        
        if mask:
            img1 = color.getColorMask(img1,'b')
            img2 = color.getColorMask(img2,'b')
        
        img = np.hstack((img1,img2))        
        
        cv2.imshow("Picture",img)
        i = i + 1
        if i >= length:
            i = 0
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    playSingleVideo(path)
    #playTwoVideos(path)