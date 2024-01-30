import os
import rembg 
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog
import progressbar

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