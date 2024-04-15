import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from skimage.feature.texture import graycomatrix, graycoprops

def brown_edge_detection(image_path):
    
    original_image = cv2.imread(image_path)

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

    # Combine brown and yellow masks
    combined_mask = cv2.bitwise_or(brown_mask, yellow_mask)

    # Create color image with the original color where brown and yellow pixels are detected
    combined_color_image = original_image.copy()
    combined_color_image[combined_mask != 0] = [0, 255, 255]  # Set combined pixels to yellow for visualization

    # Apply binary thresholding on the original image to separate the leaf from the shadow
    _, thresholded_image = cv2.threshold(combined_mask, 40, 255, cv2.THRESH_BINARY_INV)

    # Convert the binary mask to a 3-channel image
    thresholded_image_color = cv2.merge([thresholded_image, thresholded_image, thresholded_image])

    # Exclude green pixels from the original image using the binary mask
    final_image = cv2.bitwise_and(original_image, cv2.bitwise_not(thresholded_image_color))

    # Compute GLCM
    gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    distances = [1, 2, 3]  # Define distances for GLCM
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Define angles for GLCM
    glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)

    # Compute texture features from GLCM
    glcm_props = graycoprops(glcm)
    
    # Display texture features
    for prop_name, prop_values in glcm_props.items():
        print(f"{prop_name}: {prop_values}")

    root = tk.Tk()
    root.title("Image Analysis Results")

    # Create a frame for displaying the results
    result_frame = ttk.Frame(root)
    result_frame.pack(padx=10, pady=10)

    # Display unhealthy and healthy percentages
    unhealthy_label = ttk.Label(result_frame, text=f"Unhealthy: {unhealthy_percentage:.1f}%")
    unhealthy_label.pack()

    healthy_label = ttk.Label(result_frame, text=f"Healthy: {healthy_percentage:.1f}%")
    healthy_label.pack()

    # Display texture features in a table
    texture_label = ttk.Label(result_frame, text="Texture Features")
    texture_label.pack()

    texture_table = ttk.Treeview(result_frame, columns=("Metric", "Value"), show="headings")
    texture_table.heading("Metric", text="Metric")
    texture_table.heading("Value", text="Value")
    texture_table.pack()

    for prop_name, prop_values in glcm_props.items():
        texture_table.insert("", "end", values=(prop_name, np.mean(prop_values)))

    # Display images
    processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image_pil = Image.fromarray(processed_image_rgb)
    processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

    processed_label = tk.Label(root, image=processed_image_tk)
    processed_label.image = processed_image_tk
    processed_label.pack()

    extracted_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    extracted_image_pil = Image.fromarray(extracted_image_rgb)
    extracted_image_tk = ImageTk.PhotoImage(extracted_image_pil)

    extracted_label = tk.Label(result_frame, image=extracted_image_tk)
    extracted_label.image = extracted_image_tk
    extracted_label.pack()

    root.mainloop()

# Example usage
input_image = r"C:\Users\Richarde\Dropbox\YEAR 3\SEM 2\INFO 3604\Images\739ae1b1-8b4c-4e2d-9157-0424ac2f76b3.png"
brown_edge_detection(input_image)
