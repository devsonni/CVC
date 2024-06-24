import cv2 as cv
import numpy as np

def main(sift):
    vid = cv.VideoCapture(0)
    out = False
    track_img = cv.imread("tracking.jpg")
    track_img = cv.resize(track_img, (track_img.shape[1]//8, track_img.shape[0]//8))
    
    while True:
        ret, frame = vid.read()
        
        if not ret:
            break

        # Initiate SIFT detector
        sift = cv.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(track_img, None)
        kp2, des2 = sift.detectAndCompute(frame, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Use RANSAC to estimate a robust homography
        if len(good) > 4:
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
        else:
            matchesMask = None

        # Draw matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv.drawMatches(track_img, kp1, frame, kp2, good, None, **draw_params)

        # Key reading
        key = cv.waitKey(1)

        # Show the frame
        cv.imshow("Photo Booth", img3)

        # Exit on 'Esc' key press
        if key == 27:
            break

    # Release resources
    vid.release()
    if out:
        out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # Initialize SIFT detector
    sift = cv.SIFT_create()
    
    main(sift)
