import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class ImageMatcher:
    def __init__(self, img1_path, img2_path):
        """
        Initialize the ImageMatcher with paths to the two images.
        Create SIFT and SURF detectors.
        """
        self.img1 = cv.imread(img1_path, 0)  # Load the first image in grayscale
        self.img2 = cv.imread(img2_path, 0)  # Load the second image in grayscale
        
        # Initialize SIFT detector
        self.sift = cv.SIFT_create()  
        # Initialize SURF detector
        self.surf = cv.xfeatures2d_SURF.create(hessianThreshold=400)  
        
        # Basic parameters for FLANN
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=30)  # Adjust the number of checks for a trade-off between speed and accuracy
        
        # Create results directory if it doesn't exist
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def detect_and_compute(self, method='sift', img=None):
        """
        Detect keypoints and compute descriptors using the specified method.
        """
        if img is None:
            img = self.img1
            
        if method == 'sift':
            detector = self.sift
        elif method == 'surf':
            detector = self.surf
        else:
            raise ValueError("Method not recognized. Use 'sift' or 'surf'.")
        
        kp, des = detector.detectAndCompute(img, None)
        return kp, des

    def match_flann(self, des1, des2):
        """
        Match descriptors using the FLANN-based matcher.
        """
        flann = cv.FlannBasedMatcher(self.index_params, self.search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)

        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance:
                matches_mask[i] = [1, 0]

        return matches, matches_mask

    def match_bf(self, des1, des2, ratio_thresh=0.6):
        """
        Match descriptors using the BFMatcher with a ratio test.
        """
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_thresh * n.distance:
                matches_mask[i] = [1, 0]

        return matches, matches_mask

    def draw_matches(self, kp1, kp2, matches, matches_mask, method_name):
        """
        Draw matches on the images and display the result.
        """
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=cv.DrawMatchesFlags_DEFAULT)
        result_img = cv.drawMatchesKnn(self.img1, kp1, self.img2, kp2, matches, None, **draw_params)
        
        # Save the result in the results folder
        cv.imwrite(os.path.join(self.results_dir, f"{method_name}.jpg"), cv.cvtColor(result_img, cv.COLOR_RGB2BGR))

        return result_img

    def save_subplot(self, images, titles):
        """
        Save a subplot with images and titles.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for ax, img, title in zip(axes.ravel(), images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        
        # Save the subplot in the results folder
        plt.savefig(os.path.join(self.results_dir, 'matches_subplot.jpg'))
        plt.show()

    def run(self):
        """
        Run the matching process for both SIFT and SURF using both FLANN and BF matchers.
        Measure the time taken for each process and print the results.
        """
        # Store results and titles for subplot
        results = []
        titles = []

        # SIFT with FLANN
        start_time = time.time()
        kp1, des1 = self.detect_and_compute(method='sift')
        kp2, des2 = self.detect_and_compute(method='sift', img=self.img2)
        matches, matches_mask = self.match_flann(des1, des2)
        res_sift_flann = self.draw_matches(kp1, kp2, matches, matches_mask, 'SIFT_FLANN')
        sift_flann_time = time.time() - start_time
        print(f"SIFT with FLANN took {sift_flann_time:.2f} seconds")
        results.append(res_sift_flann)
        titles.append('SIFT with FLANN')

        # SIFT with BF
        start_time = time.time()
        kp1, des1 = self.detect_and_compute(method='sift')
        kp2, des2 = self.detect_and_compute(method='sift', img=self.img2)
        matches_bf, matches_mask_bf = self.match_bf(des1, des2, ratio_thresh=0.6)
        res_sift_bf = self.draw_matches(kp1, kp2, matches_bf, matches_mask_bf, 'SIFT_BF')
        sift_bf_time = time.time() - start_time
        print(f"SIFT with BF took {sift_bf_time:.2f} seconds")
        results.append(res_sift_bf)
        titles.append('SIFT with BF')

        # SURF with FLANN
        start_time = time.time()
        kp1, des1 = self.detect_and_compute(method='surf')
        kp2, des2 = self.detect_and_compute(method='surf', img=self.img2)
        matches_surf, matches_mask_surf = self.match_flann(des1, des2)
        res_surf_flann = self.draw_matches(kp1, kp2, matches_surf, matches_mask_surf, 'SURF_FLANN')
        surf_flann_time = time.time() - start_time
        print(f"SURF with FLANN took {surf_flann_time:.2f} seconds")
        results.append(res_surf_flann)
        titles.append('SURF with FLANN')

        # SURF with BF
        start_time = time.time()
        kp1, des1 = self.detect_and_compute(method='surf')
        kp2, des2 = self.detect_and_compute(method='surf', img=self.img2)
        matches_bf_surf, matches_mask_bf_surf = self.match_bf(des1, des2, ratio_thresh=0.6)
        res_surf_bf = self.draw_matches(kp1, kp2, matches_bf_surf, matches_mask_bf_surf, 'SURF_BF')
        surf_bf_time = time.time() - start_time
        print(f"SURF with BF took {surf_bf_time:.2f} seconds")
        results.append(res_surf_bf)
        titles.append('SURF with BF')

        # Save subplot with all four results
        self.save_subplot(results, titles)

if __name__ == "__main__":
    matcher = ImageMatcher('book.jpg', 'table.jpg')
    matcher.run()

