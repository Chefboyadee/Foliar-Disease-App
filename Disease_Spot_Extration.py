import cv2
import numpy as np
import os

def brown_edge_detection(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Read the image
        image_path = os.path.join(input_folder, file)
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

        # Combine brown and yellow masks
        combined_mask = cv2.bitwise_or(brown_mask, yellow_mask)

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

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, final_image)

    print("Processing complete. Check the output folder for the results.")

# Example usage
input_folder = 'extracted/d3/Dataset3/yellow_leaf_curl2'
output_folder = 'extracted/d3/Dataset3/yellow_leaf_curls'
brown_edge_detection(input_folder, output_folder)
