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
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary_threshold = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assumes the leaf is the largest object)
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the leaf
    leaf_mask = np.zeros_like(gray)
    cv2.drawContours(leaf_mask, [max_contour], 0, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=leaf_mask)

    return result



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
