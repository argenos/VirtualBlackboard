# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:01:36 2014

@author: Argen
"""

import numpy as np
import cv2

cap0 = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(1)

while(True):
    #print "Video capture..."
    # Capture frame-by-frame
    ret0, frame0 = cap0.read()
    #ret1, frame1 = cap1.read()
    
    newFrame0 = cv2.resize(frame0,(640,480))
    #newFrame1 = cv2.resize(frame1,(640,480))
    
    cv2.imshow('frame0',newFrame0)
    #cv2.imshow('frame1',newFrame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap0.release()
#cap1.release()
cv2.destroyAllWindows()