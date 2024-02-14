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

    # Invert the brown mask
    not_brown_mask = cv2.bitwise_not(brown_mask)

    # Exclude black pixels in the not_brown_mask
    not_brown_mask[(original_image == [0, 0, 0]).all(axis=2)] = 0

    # Calculate the percentage of brown and not brown pixels
    total_pixels = original_image.shape[0] * original_image.shape[1]
    brown_percentage = (np.count_nonzero(brown_mask) / total_pixels) * 100
    not_brown_percentage = 100 - brown_percentage


    print(f"Percentage of brown pixels: {brown_percentage:.2f}%")
    print(f"Percentage of not brown pixels: {not_brown_percentage:.2f}%")

    # Perform edge detection on the brown mask using the Canny edge detector
    edges = cv2.Canny(brown_mask, 100, 150)
    edges2 = cv2.Canny(not_brown_mask, 100, 150)

    # Create an image with the original color where brown pixels are detected
    brown_color_image = original_image.copy()
    brown_color_image[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to green for visualization

    # Create an image with green color where pixels are not brown (excluding black)
    green_color_image = original_image.copy()
    green_color_image[not_brown_mask != 0] = [0, 255, 0]  # Set non-brown pixels to green

    

    # Display the original image, the detected edges, the brown mask, and the result
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Brown Mask', cv2.cvtColor(brown_mask, cv2.COLOR_GRAY2BGR))
    cv2.imshow('Brown Color Image', brown_color_image)
    cv2.imshow('idk', cv2.cvtColor(not_brown_mask, cv2.COLOR_GRAY2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/Bacterial_spot/0ac8c80f-6d67-46ee-b662-8265d9df9183___GCREC_Bact.Sp 6115_no_bg.png'
brown_edge_detection(image_path)
