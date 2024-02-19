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

        kernel = np.ones((5, 5), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate the total number of pixels
        total_pixels = original_image.shape[0] * original_image.shape[1]

        # Calculate the percentage of unhealthy pixels
        unhealthy_pixels = np.count_nonzero(brown_mask | green_mask | yellow_mask)
        unhealthy_percentage = (unhealthy_pixels / total_pixels) * 100

        # Calculate the percentage of healthy pixels
        healthy_percentage = 100 - unhealthy_percentage

        # Create a folder for the current image with healthy and unhealthy percentage information
        image_output_folder = os.path.join(output_folder, f'image_{idx}_healthy_{healthy_percentage:.2f}_unhealthy_{unhealthy_percentage:.2f}')
        os.makedirs(image_output_folder, exist_ok=True)

        # Create an empty mask for the overlay
        overlay_mask = np.zeros_like(original_image)

        # Set the brown, green, and yellow regions in the overlay mask
        overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
        overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
        overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow

       
        # Create a text file
        healthy_txt = os.path.join(image_output_folder, 'healthy_percentage.txt')

        with open(healthy_txt, 'w') as f:
            f.write("The percentage of healthy features (pixels) on this leaf is: ") 
            f.write(str(healthy_percentage))
            f.write(" %")

        # Overlay the mask on the original image
        processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)

        # Write the processed image to the output folder
        cv2.imwrite(os.path.join(image_output_folder, 'processed_image.png'), processed_image)

# Example usage
input_folder = ("Images/No_BG/late_blight") # Input folder containing images
output_folder = ("Images/No_BG/test")  # Output folder for processed images
brown_edge_detection(input_folder, output_folder)