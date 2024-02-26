import cv2
import numpy as np

def brown_edge_detection(image_path):
    # Read the image
    original_image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the brown, green, yellow color in HSV
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([24, 255, 200])

    lower_green = np.array([40, 60, 20])
    upper_green = np.array([70, 255, 200])

    lower_yellow = np.array([25, 60, 20])
    upper_yellow = np.array([39, 255, 200])

    # Create masks for the brown, green, yellow color
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Create color images with the original color where brown, green, and yellow pixels are detected
    brown_color_image = original_image.copy()
    brown_color_image[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red for visualization

    green_color_image = original_image.copy()
    green_color_image[green_mask != 0] = [0, 255, 0]  # Set green pixels to green for visualization

    yellow_color_image = original_image.copy()
    yellow_color_image[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow for visualization

    # Combine all images horizontallyoverlay_mask = np.zeros_like(original_image)
    overlay_mask = np.zeros_like(original_image)
    # Set the brown, green, and yellow regions in the overlay mask
    overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
    overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
    overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow
    
    # Combine brown and yellow masks
    combined_mask = cv2.bitwise_or(brown_mask, yellow_mask)

    # Create color image with the original color where brown and yellow pixels are detected
    combined_color_image = original_image.copy()
    combined_color_image[combined_mask != 0] = [0, 255, 255]  # Set combined pixels to yellow for visualization
    
    # Apply erosion and dilation to reduce noise in the combined mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Apply binary thresholding on the original image to separate the leaf from the shadow
    _, thresholded_image = cv2.threshold(combined_mask, 40, 255, cv2.THRESH_BINARY_INV)

    # Convert the binary mask to a 3-channel image
    thresholded_image_color = cv2.merge([thresholded_image, thresholded_image, thresholded_image])
    

    # Exclude green pixels from the original image using the binary mask
    final_image = cv2.bitwise_and(original_image, cv2.bitwise_not(thresholded_image_color))

    # Display the combined image
    cv2.imshow('Combined Images', overlay_mask)
    cv2.imshow('Original Images', original_image)
    cv2.imshow('Final', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/septoria_leaf/0a146952-538a-41e5-a422-b5d50e91771e___Matt.S_CG 6079_no_bg.png'
brown_edge_detection(image_path)
