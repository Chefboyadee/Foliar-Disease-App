import os
import rembg 
from PIL import Image
import io
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import csv
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

import joblib

# Load the trained model
model = joblib.load('Models/catboost_model.joblib')

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

def preprocess_and_extract_features(image_path):
    # Load the image
    #image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological transform to close small holes
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Perform bitwise AND with original image to get segmented leaf
    segmented = cv2.bitwise_and(image_path, image_path, mask=closed)

    # Extract shape features
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = cv2.contourArea(contours[0])
    leaf_perimeter = cv2.arcLength(contours[0], True)

    # Extract color features
    b, g, r = cv2.split(segmented)
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    std_b = np.std(b)
    std_g = np.std(g)
    std_r = np.std(r)
    

    # Convert to HSV and calculate green ratio
    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(hsv)
    green_ratio = np.sum((h >= 30) & (h <= 70)) / np.prod(hsv.shape[:2])
    non_green_ratio = 1 - green_ratio

    # Extract texture features from GLCM
    glcm = graycomatrix(smoothed, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Assemble the feature vector 'leaf_area', 'leaf_perimeter', 'b_mean', 'g_mean', 'r_mean', 'b_std', 'g_std', 'r_std', 'contrast', 'dissimilarity'
    features = [leaf_area, leaf_perimeter, mean_b, mean_g, mean_r, std_b, std_g, std_r, contrast, dissimilarity]

    return features

def extract_features_and_save_to_csv(data_dir, output_file):
    features = []
    labels = []

    for disease_folder in os.listdir(data_dir):
        disease_path = os.path.join(data_dir, disease_folder)
        if os.path.isdir(disease_path):
            for image_file in os.listdir(disease_path):
                image_path = os.path.join(disease_path, image_file)
                feature_vector = preprocess_and_extract_features(image_path)
                features.append(feature_vector)
                labels.append(disease_folder)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['leaf_area', 'leaf_perimeter', 'b_mean', 'g_mean', 'r_mean', 'b_std', 'g_std', 'r_std', 'green_ratio', 'non_green_ratio', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'classlabel'])
        for i in range(len(features)):
            writer.writerow(features[i] + [labels[i]])

# Selects a single image and applies background/shadow removal
def process_single_image():
    
    input_image_path = filedialog.askopenfilename(title='Select image')

    output_path = os.path.join(os.path.dirname(input_image_path), 'processed_image.png')

    remove_background(input_image_path, output_path)
    final_image = remove_shadow(output_path)

    # Process the image to detect brown edges
    brown_edge_result = brown_edge_detection(final_image)
    
   
    extraction_result = Extraction(final_image)
    
    features=preprocess_and_extract_features(extraction_result)
  

   
    predicted_class = model.predict(features)[0]
    print(f"Predicted class: {predicted_class}")
    

   


 
    
      


if __name__ == "__main__":
    process_single_image()
