import cv2
import numpy as np

def brown_edge_detection(image_path):
    # Read the image
    original_image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the brown color in HSV
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([30, 255, 200])

    # Create a mask for the brown color
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Perform edge detection on the brown mask using the Canny edge detector
    edges = cv2.Canny(brown_mask, 100, 150)

    # Create an image with the original color where brown pixels are detected
    brown_color_image = original_image.copy()
    brown_color_image[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to green for visualization

    # Display the original image, the detected edges, the brown mask, and the result
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Brown Edges', edges)
    cv2.imshow('Brown Mask', cv2.cvtColor(brown_mask, cv2.COLOR_GRAY2BGR))
    cv2.imshow('Brown Color Image', brown_color_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/01be0a55-7fc0-4ea8-a177-8ff4a4611a0a___RS_L.Scorch 0891_no_bg.png'
brown_edge_detection(image_path)
