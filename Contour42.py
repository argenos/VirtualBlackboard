# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 22:32:41 2015

@author: Argen
"""

import numpy as np
import cv
import cv2
import ROISelection as r
import Color as color




def init(img, img2=None):
    roi1 = r.getROI(img)
    
    if img2==None:
        roi2 = ()
        return roi1,roi2
    else:
        roi2 = r.getROI(img2)
        return roi1,roi2

def getCroppedImage(img, roi):
    x1, y1, x2, y2 = roi
    i = img[y1:y2,x1:x2]
    return i
    
def paintContour(img, t=False, c=False):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)    
    if t:    
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold",thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)    
    #cv2.drawContours(imgray, contours, -1, (255,0,0), 4)    
    i = 0
    bigArea = 0
    for cnt in contours:
       area = cv2.contourArea(cnt)
       if area>= bigArea:
           bigArea=i
       i = i + 1
       
    
    cnt = contours[i-1]
    #cv2.drawContours(imgray, [cnt], 0, (255,255,255), 3)
    
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    img = cv2.circle(imgray,center,radius,(255,0,0),2)
    
    if c:
        cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        cv2.imshow("Contours", imgray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img

def hasContour(img):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if contours ==None:
        return False
    return True

def getMarker(img):
    '''
    Receives an image in BGR
    '''
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)    
    i = 0
    bigArea = 0
    for cnt in contours:
       area = cv2.contourArea(cnt)
       if area>= bigArea:
           bigArea=i
       i = i + 1       
    
    cnt = contours[i-1]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
    return center, radius
    

def getImgWithMarker(original,roi,c):
    '''
    Receives an image cropped 
    '''
    org = original.copy()
    img = getCroppedImage(org,roi)
    imc = color.getColorMask(img,c) 
    imgray = cv2.cvtColor(imc,cv2.COLOR_HSV2BGR)
    center,radius = getMarker(imgray)
    cv2.circle(img,center,radius,(255,0,0),2)
    org[roi[1]:roi[3],roi[0]:roi[2]]=img
    
    return org
    

    
def main():
    ex = cv2.imread('images/images_azul/c2_image04.png')
    ex = cv2.GaussianBlur(ex, (3, 3), 0)
    roi1,roi2 = init(ex)
    
    x1, y1, x2, y2 = roi1
    im=getImgWithMarker(ex,roi1,'b')
    img = cv2.resize(im,(640,480))
    
    cv2.imshow("Marker detection",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()