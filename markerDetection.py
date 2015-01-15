__author__ = 'laverden'


import numpy as np
import cv2 as cv
import glob
import Color as Color
import circleDetection as Circle
import ROISelection as Cropper
import projection as Artist
import matplotlib.pyplot as plt


def diff_frame(frame1, frame2, display):
    im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    im1 = cv.GaussianBlur(im1, (3, 3), sigmaY=0, sigmaX=0)
    im2 = cv.GaussianBlur(im2, (3, 3), sigmaY=0, sigmaX=0)
    diff = cv.absdiff(im1, im2)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel, iterations=2)

    if display:
        cv.namedWindow("Difference-OP", cv.WINDOW_NORMAL)
        cv.imshow("Difference-OP", np.hstack((diff, opening)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    return opening


def detect_background(frame1, frame2):
    f1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    f2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    f1 = cv.equalizeHist(f1)
    f2 = cv.equalizeHist(f2)
    f1 = cv.GaussianBlur(f1, (5, 5), 5)
    f2 = cv.GaussianBlur(f2, (5, 5), 5)
    diff = cv.absdiff(f1, f2)
    transform = cv.dilate(diff, kernel=(5, 5), iterations=10)
    transform = cv.erode(transform, kernel=(5, 5), iterations=10)
    ret, binary = cv.threshold(transform, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    closing = cv.morphologyEx(binary, kernel=(5, 5), op=cv.MORPH_DILATE, iterations=10)
    mask = cv.medianBlur(closing, 9)

    masked_im = cv.bitwise_and(frame1, frame1, mask=mask)

    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    original = np.vstack((f1, f2))
    cv.imshow("Original", original)
    cv.namedWindow("Difference-Transformation", cv.WINDOW_NORMAL)
    t = np.hstack((diff, transform))
    cv.imshow("Difference-Transformation", t)
    cv.namedWindow("Otsu", cv.WINDOW_NORMAL)
    cv.imshow("Otsu", binary)
    cv.namedWindow("Closed Image and Mask", cv.WINDOW_NORMAL)
    masking = np.hstack((closing, mask))
    cv.imshow("Closed Image and Mask", masking)
    cv.namedWindow("Masked Image", cv.WINDOW_NORMAL)
    cv.imshow("Masked Image", masked_im)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return masked_im


def get_corners(display):
    cam1 = sorted(np.array(glob.glob("images/setup/corners/c1_corner*.png")))
    cam2 = sorted(np.array(glob.glob("images/setup/corners/c2_corner*.png")))
    bck_g1 = cv.imread("images/setup/background/background1.png")
    bck_g2 = cv.imread("images/setup/background/background2.png")

    corners_1 = []
    corners_2 = []
    lim1 = [0,0,985,bck_g1.shape[0]]
    lim2 = [0,740,bck_g2.shape[1], 990]

    for f1, f2 in zip(cam1, cam2):
        im1 = cv.imread(f1)
        im2 = cv.imread(f2)
        diff1 = diff_frame(bck_g1, im1, display=False)
        diff2 = diff_frame(bck_g2, im2, display=False)
        ret1, binary1 = cv.threshold(diff1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        ret2, binary2 = cv.threshold(diff2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        ret1, c1 = circles_by_contour(binary1, None, lim1, auto_threshold=False, full_display=False)
        ret2, c2 = circles_by_contour(binary2, None, lim2, auto_threshold=False, full_display=False)
        Circle.draw_one_circle(im1, c1)
        Circle.draw_one_circle(im2, c2)
        corners_1.append(c1[0:2])
        corners_2.append(c2[0:2])
        if display:
            cv.namedWindow("Images", cv.WINDOW_NORMAL)
            cv.imshow("Images", np.hstack((im1, im2)))
            cv.namedWindow("Binary", cv.WINDOW_NORMAL)
            cv.imshow("Binary", np.hstack((binary1, binary2)))
            cv.waitKey(0)
            cv.destroyAllWindows()

    if display:
        print "Corners SideView:\n", corners_1
        print "\nCorners TopView:\n", corners_2

    return corners_1, corners_2


def refine_roi(corner1, corner2):
    d = 30
    roi1 = np.array([corner1[0][0]-d, corner1[0][1]-d, corner1[3][0]+d, corner1[2][1]+d])
    roi2 = np.array([corner2[3][0]-d, corner2[1][1]-d, corner2[0][0]+d, corner2[0][1]+d])
    return roi1, roi2


def circles_by_contour(image, c, roi, auto_threshold, full_display):
    if auto_threshold:
        im = cv.GaussianBlur(image, (9, 9), 0)

        copy = im.copy()
        cropped = im.copy()
        cropped = cropped[roi[1]:roi[3], roi[0]:roi[2]]

        color_mask = Color.color_mask(cropped, color=c, display=full_display)
        gray = cv.cvtColor(color_mask, cv.COLOR_HSV2BGR)
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 3)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        eros = cv.erode(thresh, (7, 7), 30)
        #thresh = cv.dilate(eros, (5,5), 30)

        if full_display:
            cv.namedWindow("Erosion-Dilatation", cv.WINDOW_NORMAL)
            cv.imshow("Erosion-Dilatation", np.hstack((eros, thresh)))
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        copy = image.copy()
        cropped = image.copy()
        cropped = cropped[roi[1]:roi[3], roi[0]:roi[2]]
        thresh = cropped

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i = 0
    index = -1
    biggest_area = 20

    for cnt in contours:
        area = cv.contourArea(cnt)
        if biggest_area <= area:
            biggest_area = area
            index = i
        i += 1
    #print "[%s] = %d" % (i, biggest_area)
    if index >= 0:
        (x_c, y_c), radius = cv.minEnclosingCircle(contours[index])
    else:
        x_c = y_c = 0
        radius = 0
        #print "No CIRCLE"

    # Relative Circle
    center = (int(x_c), int(y_c))
    radius = int(radius)
    # Absolute Center
    center = (int(x_c) + roi[0], int(y_c) + roi[1])

    if radius != 0:
        cv.circle(copy, center, 1, (10, 255, 25), 5)
        cv.circle(copy, center, 15, (10, 10, 255), 2)
        #print "Center: ", center_abs

    param = np.array([center[0], center[1], radius])

    return copy, param


def main():
    print "VIRTUAL BOARD\n"
    i = 10
    dir_name = "images/images_verde/"
    color_ball = 'g'
    # 1 for color-mask, 2 for background-segmentation
    detector = 2
    roi_method = 'auto'
    files1 = sorted(np.array(glob.glob(dir_name + "c1_image*.png")))
    files2 = sorted(np.array(glob.glob(dir_name + "c2_image*.png")))
    bck_g1 = cv.imread("images/setup/background/background1.png")
    bck_g2 = cv.imread("images/setup/background/background2.png")

    print "Directory: ", dir_name
    print "Color: ", color_ball
    print "Detector Method: ", detector
    print "ROI Method: ", roi_method
    ###################################################################
    ## CONTOUR METHOD APPLYING COLOR MASK OVER ALL FRAMES
    ###################################################################
    total_frames = len(files1)
    frame_number = 0

    if roi_method == 'auto':
        cor1, cor2 = get_corners(display=False)
        roi1_xy, roi2_xy = refine_roi(cor1, cor2)
    else:
        cal_frame1 = cv.imread(files1[0])
        cal_frame2 = cv.imread(files2[0])
        # WHAT: roi_xy = [x1, y1, x2, y2]
        roi1_xy = np.array([0, 0, cal_frame1.shape[1], cal_frame1.shape[0]])
        roi1_xy[0], roi1_xy[1], roi1_xy[2], roi1_xy[3] = Cropper.getROI(cal_frame1)
        roi2_xy = np.array([0, 0, cal_frame2.shape[1], cal_frame2.shape[0]])
        roi2_xy[0], roi2_xy[1], roi2_xy[2], roi2_xy[3] = Cropper.getROI(cal_frame2)

    cv.namedWindow("MarkerSide", cv.WINDOW_NORMAL)
    cv.namedWindow("MarkerTop", cv.WINDOW_NORMAL)
    cv.namedWindow("Canvas", cv.WINDOW_NORMAL)
    Artist.init()
    plt.figure()
    while frame_number < total_frames:
        if detector == 1:
            c, center = circles_by_contour(cv.imread(files1[frame_number]), c=color_ball, roi=roi1_xy,
                                           auto_threshold=True, full_display=False)
            c2, center2 = circles_by_contour(cv.imread(files2[frame_number]), c=color_ball, roi=roi2_xy,
                                             auto_threshold=True, full_display=False)
        else:
            diff1 = diff_frame(bck_g1, cv.imread(files1[frame_number]), display=False)
            diff2 = diff_frame(bck_g2, cv.imread(files2[frame_number]), display=False)
            ret1, binary1 = cv.threshold(diff1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            ret1, binary2 = cv.threshold(diff2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            r1, center = circles_by_contour(binary1, None, roi1_xy, auto_threshold=False, full_display=False)
            r2, center2 = circles_by_contour(binary2, None, roi2_xy, auto_threshold=False, full_display=False)
            c = cv.imread(files1[frame_number])
            c2 = cv.imread(files2[frame_number])
            Circle.draw_one_circle(c, center)
            Circle.draw_one_circle(c2, center2)

        virtual, point = Artist.project_on_board(center[1], center2[0], color_ball)
        cv.imshow("MarkerSide", c)
        cv.imshow("MarkerTop", c2)
        #cv.imshow("Canvas", virtual)

        #plt.scatter(center2[0], -center2[1], c='r')
        #plt.scatter(center[0], -center[1], c='g')
        plt.scatter(point[0], -point[1]+virtual.shape[0], c=color_ball)
        #plt.scatter(center2[0], -center[1], c='b')
        k = cv.waitKey(10) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break
        frame_number += 1
    cv.waitKey(10)

    cv.destroyAllWindows()
    #cv.imwrite("images/results/result%d_board.jpg" % i, virtual)
    plt.xlim([0, 2000])
    plt.ylim([0, 1125])
    txt = "\ndetec=%d, roi=%s" % (detector, roi_method)
    plt.title(dir_name+txt)
    #plt.savefig('images/results/result%d_plot.jpg' % i)
    plt.show()



if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
