import os
import rembg 
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog
import progressbar
import cv2
import numpy as np
# Removes background from image and saves output to given path
def remove_background(input_path, output_path):

  # Open image and read data
  with open(input_path, "rb") as input_file:
    input_data = input_file.read()
  
  # Use rembg to remove background    
  output_data = rembg.remove(input_data)
  
  # Open the image again from the modified data
  output_image = Image.open(io.BytesIO(output_data))

  # Save the foreground image to output path
  output_image.save(output_path)

def remove_shadow(image_path):
   # Read the image
    original_image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the color range for the green color of the leaf
    lower_green = np.array([5, 0, 40])  # Adjust these values based on your specific image
    upper_green = np.array([90, 255, 255])  # Adjust these values based on your specific image

    # Threshold the image to obtain a binary mask for the green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply binary thresholding on the original image to separate the leaf from the shadow
    _, thresholded_image = cv2.threshold(green_mask, 40, 255, cv2.THRESH_BINARY)

    # Use a median blur to reduce noise
    thresholded_image = cv2.medianBlur(thresholded_image, 5)

    # Convert the binary mask to a 3-channel image
    thresholded_image_color = cv2.merge([thresholded_image, thresholded_image, thresholded_image])

    # Retain only the leaf in the original image using the binary mask
    final_image = cv2.bitwise_and(original_image, thresholded_image_color)

    return final_image



# Goes through all files in input folder, removes background, 
# and saves to output folder
def process_images(input_folder, output_folder):

  # Create output folder if doesn't exist
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Loop through all files in input folder
  for i, filename in enumerate(os.listdir(input_folder)):

    # Only process JPG and jpeg images
    if filename.endswith(".JPG") or filename.endswith(".jpeg"):

      # Get full input path 
      input_path = os.path.join(input_folder, filename)
      
      # Generate output path in output folder
      output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_no_bg.png")

      # Call remove_background on this file
      remove_background(input_path, output_path)
      
      leaf_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_no_bg.png")
      leaf_processed_image = remove_shadow(leaf_image_path)

      # Save the processed leaf image
      cv2.imwrite(leaf_image_path, leaf_processed_image)


      # Print filename to show progress  
      print(f"Processed: {filename}")

      # Update progress bar
      progbar.update(i+1)
      

# Main function
if __name__ == "__main__":

  # Use tkinter filedialog to select folders
  root = tk.Tk()
  root.withdraw()

  input_folder_path = filedialog.askdirectory(title='Select input folder')
  output_folder_path = filedialog.askdirectory(title='Select output folder')

  # Total files to process
  total_files = len(os.listdir(input_folder_path))

  # Create progress bar
  progbar = progressbar.ProgressBar(maxval=total_files)
  progbar.start()

  # Prints when processing starts  
  print("Starting image processing...")

  # Call processing function
  process_images(input_folder_path, output_folder_path)

  # Prints when done
  print("Processing complete!")

  # Close progress bar
  progbar.finish()