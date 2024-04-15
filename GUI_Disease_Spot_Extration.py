import os
import rembg 
from PIL import Image
import io
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Removes background from image and saves output to given path
def remove_background(input_path, output_path):
    with open(input_path, "rb") as input_file:
        input_data = input_file.read()
    output_data = rembg.remove(input_data)
    output_image = Image.open(io.BytesIO(output_data))
    output_image.save(output_path)

def remove_shadow(image_path):
    original_image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([5, 0, 40])  # Adjust these values as needed
    upper_green = np.array([90, 255, 255])  # Adjust these values as needed
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    _, thresholded_image = cv2.threshold(green_mask, 40, 255, cv2.THRESH_BINARY)
    thresholded_image = cv2.medianBlur(thresholded_image, 5)
    thresholded_image_color = cv2.merge([thresholded_image, thresholded_image, thresholded_image])
    final_image = cv2.bitwise_and(original_image, thresholded_image_color)
    return final_image

def brown_edge_detection(original_image):
        
   
        

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the brown, green, and yellow colors in HSV
        lower_brown = np.array([10, 60, 20])
        upper_brown = np.array([24, 255, 200])

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

        # Calculate the total number of pixels
        total_pixels = original_image.shape[0] * original_image.shape[1]

        # Calculate the percentage of unhealthy pixels (brown and yellow)
        unhealthy_pixels = np.count_nonzero(brown_mask | yellow_mask)
        unhealthy_percentage = (unhealthy_pixels / total_pixels) * 100

        # Calculate the percentage of healthy pixels (green)
        healthy_pixels = np.count_nonzero(green_mask)
        healthy_percentage = (healthy_pixels / total_pixels) * 100

        total_percentage = unhealthy_percentage + healthy_percentage
        unhealthy_percentage = (unhealthy_percentage / total_percentage) * 100
        healthy_percentage = (healthy_percentage / total_percentage) * 100

        # Create an empty mask for the overlay
        overlay_mask = np.zeros_like(original_image)

        # Set the brown, green, and yellow regions in the overlay mask
        overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
        overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
        overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow


        processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)

        return {'processed_image': processed_image, 'unhealthy_percentage': unhealthy_percentage,'healthy_percentage': healthy_percentage}

def Extraction(original_image):
    

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

    return final_image



# Selects a single image and applies background/shadow removal
def process_single_image():
    root = tk.Tk()
    root.title("Image Analysis Results")

    input_image_path = filedialog.askopenfilename(title='Select image')

    output_path = os.path.join(os.path.dirname(input_image_path), 'processed_image.png')

    remove_background(input_image_path, output_path)
    final_image = remove_shadow(output_path)

    # Process the image to detect brown edges
    brown_edge_result = brown_edge_detection(final_image)
    
    # Assuming you have a function named 'Extraction' that processes 'final_image'
    extraction_result = Extraction(final_image)

    # Display processed image
    processed_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    processed_image_pil = Image.fromarray(processed_image_rgb)
    processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

    processed_label = tk.Label(root, image=processed_image_tk)
    processed_label.image = processed_image_tk
    processed_label.grid(row=0, column=0, padx=10, pady=10)

    # Display brown edge image
    brown_edge_rgb = cv2.cvtColor(brown_edge_result['processed_image'], cv2.COLOR_BGR2RGB)
    brown_edge_pil = Image.fromarray(brown_edge_rgb)
    brown_edge_tk = ImageTk.PhotoImage(brown_edge_pil)

    brown_edge_label = tk.Label(root, image=brown_edge_tk)
    brown_edge_label.image = brown_edge_tk
    brown_edge_label.grid(row=0, column=1, padx=10, pady=10)

    # Display unhealthy percentage below the brown edge image
    unhealthy_percentage_label = tk.Label(root, text=f"Unhealthy: {brown_edge_result['unhealthy_percentage']:.1f}%")
    unhealthy_percentage_label.grid(row=1, column=0, columnspan=2, pady=5)

    # Display healthy percentage below the brown edge image
    healthy_percentage_label = tk.Label(root, text=f"Healthy: {100 - brown_edge_result['unhealthy_percentage']:.1f}%")
    healthy_percentage_label.grid(row=2, column=0, columnspan=2, pady=5)

    # Display extraction image
    extraction_image_rgb = cv2.cvtColor(extraction_result, cv2.COLOR_BGR2RGB)
    extraction_image_pil = Image.fromarray(extraction_image_rgb)
    extraction_image_tk = ImageTk.PhotoImage(extraction_image_pil)

    extraction_label = tk.Label(root, image=extraction_image_tk)
    extraction_label.image = extraction_image_tk
    extraction_label.grid(row=3, column=0, padx=10, pady=10)

    root.mainloop()
    
    
    
      


if __name__ == "__main__":
    process_single_image()
