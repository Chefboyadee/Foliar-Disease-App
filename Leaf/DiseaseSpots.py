import cv2
import numpy as np

def brown_edge_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the brown color in HSV
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([30, 255, 200])

    # Create a mask for the brown color
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Perform edge detection on the brown mask using the Canny edge detector
    edges = cv2.Canny(brown_mask, 100, 150)

    # Convert edges to three channels to make it compatible with the original image
    colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine the original image with the colored edges
    result = cv2.addWeighted(image, 1, colored_edges, 1, 0)

    # Display the original image, the detected edges, and the result
    cv2.imshow('Original Image', image)
    cv2.imshow('Brown Edges', edges)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/01be0a55-7fc0-4ea8-a177-8ff4a4611a0a___RS_L.Scorch 0891_no_bg.png'
brown_edge_detection(image_path)
