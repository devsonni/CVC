import cv2
import numpy as np
import matplotlib.pyplot as plt

class VanishingPointDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.color_image = cv2.imread(image_path)
        self.processed_image = self.color_image.copy()
        self.gray_image = self.convert_to_gray(self.color_image)
        self.blurred_image = self.apply_gaussian_blur(self.gray_image)
        self.edges = self.detect_edges(self.blurred_image)
        self.lines = self.detect_lines(self.edges)
        self.filtered_lines = self.filter_lines(self.lines)
        self.vanishing_point = self.calculate_vanishing_point(self.filtered_lines)
    
    def convert_to_gray(self, color_image):
        """Convert the image to grayscale."""
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    def apply_gaussian_blur(self, gray_image):
        """Apply Gaussian blur to the image."""
        return cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    def detect_edges(self, blurred_image):
        """Detect edges using Canny edge detector."""
        return cv2.Canny(blurred_image, 300, 200)
    
    def detect_lines(self, edges):
        """Detect lines using Hough transform."""
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        return lines
    
    def filter_lines(self, lines):
        """Filter lines based on their angle."""
        filtered_lines = []
        for line in lines:
            rho, theta = line[0]
            if (0.4 < theta < 1.47) or (1.67 < theta < 2.74):
                filtered_lines.append((rho, theta))
        return filtered_lines
    
    def visualize_lines(self, lines, image):
        """Visualize the filtered lines on the image."""
        const = 2000
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + const * (-b))
            y1 = int(y0 + const * (a))
            x2 = int(x0 - const * (-b))
            y2 = int(y0 - const * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    def calculate_vanishing_point(self, lines):
        """Calculate the vanishing point using least squares method."""
        A = []
        B = []
        
        for rho, theta in lines:
            A.append([np.cos(theta), np.sin(theta)])
            B.append([rho])
        
        A = np.array(A)
        B = np.array(B)
        
        # least_sqrs
        ATA_inv = np.linalg.inv(A.T @ A)
        vanishing_point = ATA_inv @ A.T @ B
        
        return vanishing_point
    
    def display_images(self):
        """Display the edge-detected image, the image with lines, and the final image with the vanishing point in one subplot."""
        final_image = self.color_image.copy()
        cv2.circle(final_image, (int(self.vanishing_point[0]), int(self.vanishing_point[1])), 20, (0, 0, 255), -1)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        
        self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.edges, cmap='gray')
        axes[0].set_title("Edges")
        axes[0].axis('off')
        
        axes[1].imshow(self.processed_image)
        axes[1].set_title("Lines")
        axes[1].axis('off')
        
        axes[2].imshow(final_image)
        axes[2].set_title("Vanishing Point")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('vanishing_point.png')
        plt.show()
        

# Usage
detector = VanishingPointDetector('scene3.jpg')
detector.visualize_lines(detector.filtered_lines, detector.processed_image)
detector.display_images()
