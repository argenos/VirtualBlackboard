__author__ = 'Argen'

import ROI42 as roi
import cv2
import numpy as np

up = []
lower = []
_x = _y = 0
_im = im_copy = original = None
_clicked = done = False
clicks = 0
_xc = _yc = 0
_rad = 0


def init(img):
    cv2.namedWindow('HSV',cv2.WINDOW_NORMAL)
    #print "Please select a small area containing the marker."
    im = img.copy()
    #im = cv2.resize(im,(300,300))
    cv2.imshow('HSV',im)
    copy = im.copy()
    cv2.imshow("Selected region", copy)
    return img, im, copy


def lowerbound(x,y):
    global im_copy,lower
    lower = im_copy[y,x]
    cv2.circle(im_copy,(x,y),1,(255,255,255),1)
    cv2.imshow('Selected region',im_copy)
    return lower


def upperbound(x,y):
    global im_copy,up
    up = im_copy[y,x]
    cv2.circle(im_copy,(x,y),1,(255,255,255),1)
    cv2.imshow('Selected region',im_copy)
    return up


def computeRadius(x,y):
    global _rad, _xc, _yc
    _rad = np.uint8(np.sqrt((_xc-x)**2 + (_yc-y)**2))


def paintCircle():
    global im_copy
    cv2.circle(im_copy,(_xc,_yc),_rad,(0,0,255),-1)
    cv2.imshow("Selected region",im_copy)


def getCirclePixels():
    global im_copy, _xc, _yc, _rad
    mask = np.zeros(im_copy.shape,np.uint8)
    cv2.circle(mask,(_xc,_yc),_rad,(255,255,255),-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    cv2.imshow("Mask",mask)
    #print 'Image copy',im_copy.shape
    #print 'Mask',mask.shape
    #print 'Pixels',pixelpoints.shape

    #test = cv2.bitwise_and(im_copy,im_copy,mask=pixelpoints)
    #cv2.imshow("Test",test)
    return pixelpoints


def inCircle(x,y):
    global _xc, _yc, _rad
    dist = np.uint8(np.sqrt((_xc-x)**2 + (_yc-y)**2))
    #print dist
    if dist <= _rad:
        return True
    else:
        return False


def findBoundaries(img):
    global im_copy
    '''
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    '''
    copy = img.copy()
    copy = cv2.GaussianBlur(copy, (5, 5), 0)
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    pixels = getCirclePixels()
    #hsv = cv2.medianBlur(hsv,3)
    minH = 179
    minS = 255
    minV = 255
    maxH = 0
    maxS = 0
    maxV = 0
    #print hsv.shape

    for k in xrange(pixels.shape[0]):
        #for x in xrange(pixels.shape[1]):
            #if inCircle(x,y):
        #cv2.circle(im_copy,(pixels[k, 0],pixels[k, 0]),1,(255,0,0),-1)
        H, S, V = hsv[pixels[k,0], pixels[k,1], :]
            #print '[%d,%d]: %d, %d, %d'%(j,i,H,S,V)
        if H > maxH:
            maxH = H
        if H < minH :
            minH = H
        if S > maxS:
            maxS = S
        if S < minS:
            minS = S
        if V > maxV:
            maxV = V
        if V < minV:
            minV = V

    lower_boundary = np.array([minH,  minS, minV], dtype="uint8")
    upper_boundary = np.array([maxH,  maxS, maxV], dtype="uint8")

    print lower_boundary, upper_boundary

    return lower_boundary, upper_boundary


def applyHSV(img, lb, ub):
    copy = img.copy()
    copy = cv2.GaussianBlur(copy, (5, 5), 0)
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lb, ub)

    res = cv2.bitwise_and(copy, copy, mask=mask)
    #res = cv2.resize(res,(640,480))
    #cv2.imshow("Result", res)

    return res


def mouseHSV(event, x, y, flags, param):
    global clicks,_x,_y, _clicked,done, _xc, _yc, _im, im_copy
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
            _xc = x
            _yc = y
            lowerbound(x,y)
        elif clicks==2:
            upperbound(x,y)
            computeRadius(x,y)
            paintCircle()
        else:
            clicks = 0
            im_copy = _im.copy()
            cv2.imshow("Selected region", im_copy)
            done = True


def getHSV(img):
    global original,_im, im_copy
    original,_im, im_copy = init(img)
    cv2.setMouseCallback('HSV',mouseHSV)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            getCirclePixels()
            l, u = findBoundaries(_im)
            applyHSV(original, l, u)
            cv2.destroyAllWindows()
            return l, u
            #cv2.imshow("Result", im_copy)

if __name__ == '__main__':
    img = cv2.imread('images/frames/blue/c1_image020.png')
    #img = cv2.resize(img,(1200,700))
    getHSV(img)
    print lower
    print up
    cv2.destroyAllWindows()

    '''
    Lower:  [83 35  0]
    Upper:  [134 140 214]
    Lower: np.array([83, 35, 0], dtype="uint8")
    Upper: np.array([134, 140, 214], dtype="uint8")
        '''