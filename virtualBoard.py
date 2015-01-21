import numpy as np
import cv2 as cv
import glob
import Color as Color
import circleDetection as Circle
import markerDetection as Detector
import ROISelection as Cropper
import projection as drawer
import matplotlib.pyplot as plt


def main():
    print "VIRTUAL BOARD\n"
    i = 1
    dir_name = "images/frames/red/"
    color_ball = 'r'
    roi_method = 'auto'
    files1 = sorted(np.array(glob.glob(dir_name + "c1_image*.png")))
    files2 = sorted(np.array(glob.glob(dir_name + "c2_image*.png")))
    bck_g1 = cv.imread("images/setup/background/background1.png")
    bck_g2 = cv.imread("images/setup/background/background2.png")

    print "Directory: ", dir_name
    print "Color: ", color_ball
    print "ROI Method: ", roi_method

    ###################################################################
    ## CONTOUR METHOD APPLYING COLOR MASK OVER ALL FRAMES
    ###################################################################
    total_frames = len(files1)
    frame_number = 0

    # ROI Generation/Selection:
    if roi_method == 'auto':
        cor1, cor2 = Detector.get_corners(display=False)
        roi1_xy, roi2_xy = Detector.refine_roi(cor1, cor2)
    else:
        cal_frame1 = cv.imread(files1[0])
        cal_frame2 = cv.imread(files2[0])
        # WHAT: roi_xy = [x1, y1, x2, y2]
        roi1_xy = np.array([0, 0, cal_frame1.shape[1], cal_frame1.shape[0]])
        roi1_xy[0], roi1_xy[1], roi1_xy[2], roi1_xy[3] = Cropper.getROI(cal_frame1)
        roi2_xy = np.array([0, 0, cal_frame2.shape[1], cal_frame2.shape[0]])
        roi2_xy[0], roi2_xy[1], roi2_xy[2], roi2_xy[3] = Cropper.getROI(cal_frame2)

    # Color Calibration using the corner images.
    color_ini1 = cv.imread("images/setup/corners/c1_corner00.png")
    color_ini2 = cv.imread("images/setup/corners/c2_corner00.png")
    color_ini1 = color_ini1[roi1_xy[1]:roi1_xy[3], roi1_xy[0]:roi1_xy[2], :]
    color_ini2 = color_ini2[roi2_xy[1]:roi2_xy[3], roi2_xy[0]:roi2_xy[2], :]
    Color.initializeBoundaries(color_ini1, color_ini2)
    # Initialization.
    cv.namedWindow("MarkerSide", cv.WINDOW_NORMAL)
    cv.namedWindow("MarkerTop", cv.WINDOW_NORMAL)
    cv.namedWindow("Canvas", cv.WINDOW_NORMAL)
    drawer.init()
    plt.figure()

    # Processing every frame in the sequence.
    while frame_number < total_frames:
        frame_c1 = cv.imread(files1[frame_number])
        frame_c2 = cv.imread(files2[frame_number])
        masked_c1, mask_c1 = Color.getAutoColorMask(frame_c1)
        masked_c2, mask_c2 = Color.getAutoColorMask(frame_c2)
        masked_bck1 = cv.bitwise_and(bck_g1, bck_g1, mask=mask_c1)
        masked_bck2 = cv.bitwise_and(bck_g2, bck_g2, mask=mask_c2)
        diff1 = Detector.diff_frame(bck_g1, frame_c1, display=False)
        diff2 = Detector.diff_frame(bck_g2, frame_c2, display=False)

        ret1, binary1 = cv.threshold(diff1, 50, 255, cv.THRESH_BINARY)
        ret2, binary2 = cv.threshold(diff2, 50, 255, cv.THRESH_BINARY)

        c1, center1 = Detector.circles_by_contour(binary1, c=color_ball, roi=roi1_xy,
                                                  auto_threshold=False, full_display=False)
        c2, center2 = Detector.circles_by_contour(binary2, c=color_ball, roi=roi2_xy,
                                                  auto_threshold=False, full_display=False)

        virtual, point = drawer.project_on_board(center1[1], center2[0], color_ball)
        cv.imshow("MarkerSide", binary1)
        cv.imshow("MarkerTop", binary2)
        cv.imshow("Canvas", virtual)

        plt.scatter(point[0], -point[1]+virtual.shape[0], c=color_ball)
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
    txt = "AUTO-COLOR MASKING + BACKGROUND DIFFERENCE (%d)\n" % i
    plt.title(txt + dir_name)
    #plt.savefig('images/results/result%d_plot.jpg' % i)
    plt.show()


if __name__ == "__main__":
    main()
    cv.destroyAllWindows()