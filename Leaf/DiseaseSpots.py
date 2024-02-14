import cv2
import numpy as np
import os

def brown_edge_detection(input_folder, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    # Filter out only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, image_file in enumerate(image_files):
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Error: Unable to read image {image_file}")
            continue

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the brown, green, and yellow colors in HSV
        lower_brown = np.array([10, 60, 20])
        upper_brown = np.array([20, 255, 200])

        lower_green = np.array([40, 60, 20])
        upper_green = np.array([70, 255, 200])

        lower_yellow = np.array([25, 60, 20])
        upper_yellow = np.array([39, 255, 200])

        # Create masks for the brown, green, and yellow colors
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Create an empty mask for the overlay
        overlay_mask = np.zeros_like(original_image)

        # Set the brown, green, and yellow regions in the overlay mask
        overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
        overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
        overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow

        # Overlay the mask on the original image
        processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)

        # Create a folder for the current image
        image_output_folder = os.path.join(output_folder, f'image_{idx}')
        os.makedirs(image_output_folder, exist_ok=True)

        # Write the processed image to the output folder
        cv2.imwrite(os.path.join(image_output_folder, 'processed_image.png'), processed_image)

# Example usage
input_folder = 'input folder'  # Input folder containing images
output_folder = 'output folder'  # Output folder for processed images
brown_edge_detection(input_folder, output_folder)
