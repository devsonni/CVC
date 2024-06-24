import cv2
import numpy as np

def find_vanishing_point(lines):
    # Initialize variables for sums
    Sx = 0.0
    Sy = 0.0
    S = 0.0
    
    # Iterate through each pair of lines
    for i in range(len(lines)):
        rho1, theta1 = lines[i][0]
        
        for j in range(i + 1, len(lines)):
            rho2, theta2 = lines[j][0]
            
            # Check if lines are parallel (or too close to parallel)
            if np.abs(theta1 - theta2) < np.pi / 180:
                continue  # Skip parallel lines
            
            # Compute intersection of lines i and j
            if np.abs(np.sin(theta1)) < 1e-6 or np.abs(np.sin(theta2)) < 1e-6:
                continue  # Avoid division by zero for vertical lines
            
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            
            try:
                intersection = np.linalg.solve(A, b)
                
                # Accumulate values for computing average later
                Sx += intersection[0]
                Sy += intersection[1]
                S += 1
            except np.linalg.LinAlgError:
                # Singular matrix, skip this pair of lines
                continue
    
    # Calculate average intersection point (vanishing point)
    if S != 0:
        vx = Sx / S
        vy = Sy / S
        return (int(vx), int(vy))
    else:
        return None

# Load image
image = cv2.imread('texas.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny edge detector
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect lines using Hough transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

if lines is not None:
    # Find vanishing point
    vanishing_point = find_vanishing_point(lines)
    
    if vanishing_point is not None:
        # Draw vanishing point on the image
        cv2.circle(image, vanishing_point, 5, (0, 0, 255), -1)
        
        # Display the image with the vanishing point
        cv2.imshow('Vanishing Point', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No vanishing point detected.")
else:
    print("No lines detected.")
