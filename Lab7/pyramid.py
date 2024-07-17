import numpy as np
import cv2 as cv
import glob

class PyramidDrawer:
    def __init__(self):
        """Initialize the PyramidDrawer with camera matrix and distortion coefficients."""
        self.mtx = np.load('sample_data/camera_mat.npy')
        self.dist = np.load('sample_data/distorsion_mat.npy')
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        self.axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [1.5, 1.5, -3]])

    def draw_pyramid(self, img, corners, imgpts):
        """Draw the pyramid on the image."""
        imgpts = np.int32(imgpts).reshape(-1, 2)
        img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
        for i in range(4):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[4]), (255, 0, 0), 3)
        img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img

    def process_images(self):
        """Process images to detect chessboard corners and draw pyramids."""
        for i, fname in enumerate(glob.glob('sample_data/left*.jpg')):
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
            if ret:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                ret, rvecs, tvecs = cv.solvePnP(self.objp, corners2, self.mtx, self.dist)
                imgpts, jac = cv.projectPoints(self.axis, rvecs, tvecs, self.mtx, self.dist)
                img = self.draw_pyramid(img, corners2, imgpts)
                cv.imshow('img', img)
                cv.waitKey(0)
                cv.imwrite('Results/' + str(i) + '.png', img)
        cv.destroyAllWindows()

# Create an instance of the PyramidDrawer and process images
pyramid_drawer = PyramidDrawer()
pyramid_drawer.process_images()
