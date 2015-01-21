__author__ = 'laverden'

import cv2 as cv
import numpy as np

h = 1080
w = 1920
canvas = np.zeros((h, w, 3))
homo_1 = 0
homo_2 = 0
homography = 0


def init():
    global homo_1, homo_2, homography
    d = 0
    board = np.float32([[d,d], [w-d,d], [w-d,h-d], [d, h-d]])
    side = np.float32([[944,298], [957,322], [967, 578], [964,626]])
    top = np.float32([[1265,807], [658,804], [750, 773], [1203,774]])

    comb = np.float32([[1377, 294], [627, 323], [723, 579], [1268, 632]])

    homo_1, mask = cv.findHomography(side, board, cv.RANSAC, 5.0)
    homo_2, mask = cv.findHomography(top, board, cv.RANSAC, 5.0)
    homography, mask = cv.findHomography(comb, board, cv.RANSAC, 5.0)


def re_project(coordinates):
    global canvas
    point = np.float32(coordinates)
    point = point.reshape(-1, 1, 2)

    canvas_point = cv.perspectiveTransform(point, homo_2)

    print point
    print canvas_point

    #el_point = np.int32([new_p2[0,0,0], new_p1[0,0,1]])
    #el_point = el_point.reshape(-1,1,2)

    #print el_point
    #img3 = cv.polylines(img2, [np.int32(new_p1)], True, 255, 5, cv.CV_AA)
    cv.polylines(canvas, [np.int32(canvas_point)], True, [0,0,255], 5, cv.CV_AA)

    return canvas


def project_on_board(side_coord, top_coord, color):
    x = top_coord
    y = side_coord
    point = np.float32([x,y])
    point = point.reshape(-1, 1, 2)

    canvas_point = cv.perspectiveTransform(point, homography)
    if color == 'r':
        cc = [0, 0, 255]
    elif color == 'g':
        cc = [0, 255, 0]
    else:
        cc = [255, 0, 0]

    cv.polylines(canvas, [np.int32(canvas_point)], True, cc, 5, cv.CV_AA)

    return canvas, canvas_point[0][0]


def project_on_board2(side_coord, top_coord, color):
    x_s = side_coord[0]
    y_s = side_coord[1]
    x_t = top_coord[0]
    y_t = top_coord[1]
    point_side = np.float32([x_s, y_s])
    point_top = np.float32([x_t, y_t])

    point_side = point_side.reshape(-1, 1, 2)
    point_top = point_top.reshape(-1, 1, 2)

    canvas_side = cv.perspectiveTransform(point_side, homo_1)
    canvas_top = cv.perspectiveTransform(point_top, homo_2)
    canvas_point = np.array([canvas_side[0][0, 1], canvas_top[0][0, 0]])
    print "CC: ", canvas_point
    canvas_point = np.float32(canvas_point)
    canvas_point = canvas_point.reshape(-1, 1, 2)
    print "Side: ", canvas_side[0][0]
    print "Top: ", canvas_top[0][0]
    print "Canvas: ", canvas_point[0][0]

    if color == 'r':
        cc = [0, 0, 255]
    elif color == 'g':
        cc = [0, 255, 0]
    else:
        cc = [255, 0, 0]

    cv.polylines(canvas, [np.int32(canvas_point)], True, cc, 5, cv.CV_AA)

    return canvas, canvas_point[0][0]



def main():
    print "Fk"
    dir_name = "images/setup/corners/"
    img1 = cv.imread(dir_name + 'c1_corner00.png')





if __name__ == "__main__":
    main()
    cv.destroyAllWindows()