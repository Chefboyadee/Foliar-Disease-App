import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk

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





        root = tk.Tk()
        root.title("Image Analysis Results")

        unhealthy_label = tk.Label(root, text=f"Unhealthy: {unhealthy_percentage:.1f}%")
        healthy_label = tk.Label(root, text=f"Healthy: {healthy_percentage:.1f}%")

        unhealthy_label.pack()
        healthy_label.pack()
        

        # Create an empty mask for the overlay
        overlay_mask = np.zeros_like(original_image)

        # Set the brown, green, and yellow regions in the overlay mask
        overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
        overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
        overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow


        processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_pil = Image.fromarray(processed_image_rgb)
        processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

        processed_label = tk.Label(root, image=processed_image_tk)
        processed_label.image = processed_image_tk
        processed_label.pack()


        root.mainloop()




      
# Example usage
input_image = ("Images/No_BG/septoria_leaf/0a68a294-30d1-4422-ab7e-a1909ec277f7___JR_Sept.L.S 8443_no_bg.png") 
brown_edge_detection(input_image)