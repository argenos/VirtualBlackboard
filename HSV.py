__author__ = 'Argen'

import ROI42 as roi
import cv2


up = []
lower = []
_x = _y = 0
_im = None
_clicked = done = False
clicks = 0

def init(img):
    cv2.namedWindow('HSV',cv2.WINDOW_NORMAL)
    print "Please select a small area containing the marker."
    x1,y1,x2,y2 = roi.getROI(img)

    im = img[y1:y2,x1:x2]
    im = cv2.resize(im,(300,300))
    cv2.imshow('HSV',im)
    return im


def lowerbound(x,y):
    global _im,lower
    lower = _im[y,x]
    cv2.circle(_im,(x,y),1,(255,255,255),1)
    cv2.imshow('HSV',_im)
    return lower

def upperbound(x,y):
    global _im,up
    up = _im[y,x]
    cv2.circle(_im,(x,y),1,(255,255,255),1)
    cv2.imshow('HSV',_im)
    return up

def mouseHSV(event, x, y, flags, param):
    global clicks,_x,_y, _clicked,done
    #print "Event"

    if event == cv2.EVENT_LBUTTONDOWN:
        clicks = clicks + 1
        #print "Down"
        _clicked = True
    elif event == cv2.EVENT_LBUTTONUP:
        #print "up"
        _clicked = False

    if _clicked:
        if clicks == 1:
            lowerbound(x,y)
        elif clicks==2:
            upperbound(x,y)
        else:
            done = True

def getHSV(img):
    global _im
    _im = init(img)
    cv2.setMouseCallback('HSV',mouseHSV)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            cv2.destroyAllWindows()
            return up,lower

if __name__ == '__main__':
    img = cv2.imread('images/frames/blue/c2_image000.png')
    #img = cv2.resize(img,(1200,700))
    lower,up=getHSV(img)
    print lower
    print up