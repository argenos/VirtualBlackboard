__author__ = 'laverden'

import cv2 as cv
import numpy as np

h = 1040
w = 1920
canvas = np.zeros((h, w, 3))
homo_1 = 0
homo_2 = 0


def init():
    global homo_1, homo_2
    board = np.float32([[0,0], [w,0], [w,h], [0, w]])
    side = np.float32([[944,298], [957,322], [967, 578], [964,626]])
    top = np.float32([[1265,807], [658,804], [750, 773], [1203,774]])

    homo_1, mask = cv.findHomography(side, board, cv.RANSAC, 5.0)
    homo_2, mask = cv.findHomography(top, board, cv.RANSAC, 5.0)


def re_project(coordinates):
    global canvas
    p = np.float32([[(944+957+967+964)/4,(322+298+578+626)/4]])
    p2 = np.float32([[(1265+658+750+1203)/4,(807+804+733+774)/4]])
    point = np.float32(coordinates)
    point = point.reshape(-1, 1, 2)
    points_to_project = p.reshape(-1, 1, 2)
    p2 = p2.reshape(-1,1,2)

    new_p1 = cv.perspectiveTransform(points_to_project, homo_1)
    new_p2 = cv.perspectiveTransform(p2, homo_2)

    canvas_point = cv.perspectiveTransform(point, homo_2)

    print point
    print canvas_point

    #el_point = np.int32([new_p2[0,0,0], new_p1[0,0,1]])
    #el_point = el_point.reshape(-1,1,2)

    #print el_point
    #img3 = cv.polylines(img2, [np.int32(new_p1)], True, 255, 5, cv.CV_AA)
    cv.polylines(canvas, [np.int32(canvas_point)], True, [0,0,255], 5, cv.CV_AA)
    return canvas


def main():
    print "Fk"
    dir_name = "images/setup/corners/"
    img1 = cv.imread(dir_name + 'c1_corner00.png')





if __name__ == "__main__":
    main()
    cv.destroyAllWindows()