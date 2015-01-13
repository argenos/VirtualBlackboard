# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 15:32:59 2015

@author: Argen
"""
import cv2
import numpy as np

blackboard = cv2.imread('images/Blackboard.png')
_clicked = False
_x=_y=0
CURRENT_COLOR = (0,0,0)


def init():
    cv2.namedWindow('Drawing', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Drawing',480,640)


def paint(x,y,color=CURRENT_COLOR):
    cv2.circle(blackboard,(x,y),5,color,-1)

def mouse(event, x, y, flags, param):
    global _x,_y,_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        _x=x
        _y=y
        _clicked =True
    elif event == cv2.EVENT_LBUTTONUP:
        _clicked = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if _clicked:
            _x=x
            _y=y

    if _clicked == True:
        paint(_x,_y)


def main():
    init()
    cv2.setMouseCallback('Drawing', mouse)

    while True:
        cv2.imshow('Drawing',blackboard)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
