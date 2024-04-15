import cv2
import numpy as np
import os
import io
from PIL import Image as Img
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from tkinter import filedialog

# Example usage
#image_path = r"C:\Users\abdul\Desktop\Foliar research\images for pdf\0d1ab917-ea01-43d7-8a63-232bffcfbb08___Keller.St_CG 1769_no_bg.png"

def brown_edge_detection(input_folder, output_destination):
    
    # Extract folder name
    folder_name = os.path.basename(input_folder)

    # Output PDF file path
    output_pdf = os.path.join(output_destination, f"{folder_name}_output.pdf")
    # Create a PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    # Filter out only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    elements = []

    for idx, image_file in enumerate(image_files):

        # Read the image
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Error: Unable to read image {image_file}")
            continue

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the brown, green, yellow color in HSV
        lower_brown = np.array([10, 60, 20])
        upper_brown = np.array([24, 255, 200])

        lower_green = np.array([40, 40, 50])
        upper_green = np.array([60, 200, 150])

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

        #     # Convert the final image to grayscale
        # gray_final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        # # Calculate total pixels
        # total_pixels = gray_final_image.shape[0] * gray_final_image.shape[1]

        # # Count non-zero pixels in the grayscale image
        # diseased_pixels = cv2.countNonZero(gray_final_image)

        # # Calculate the percentage of diseased pixels
        # percentage_diseased = 100 - (diseased_pixels / total_pixels) * 100

        # # Create a text content for healthy percentage
        # diseased_text = f"The percentage of diseased features (pixels) on this leaf is: {percentage_diseased:.2f}%"

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

        diseased_text = f"The percentage of diseased features (pixels) on this leaf is: {unhealthy_percentage:.2f}%"
        
        # Print the percentage of diseased parts
        #print(f"Percentage of diseased parts: {unhealthy_percentage:.2f}%")

        # # Display the combined image
        # cv2.imshow('Combined Images', overlay_mask)
        # cv2.imshow('Original Images', original_image)
        # cv2.imshow('Final', final_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        # Encode processed image to PNG format
        ret, buffer = cv2.imencode('.png', overlay_mask)
        if not ret:
            print("Error: Unable to encode processed image.")
            continue

        # Create in-memory file-like object
        img_data = io.BytesIO(buffer)

        # Encode final image to PNG format
        ret, buffer = cv2.imencode('.png', final_image)
        if not ret:
            print("Error: Unable to encode final image.")
            continue

        # Create in-memory file-like object 
        final_img_data = io.BytesIO(buffer)

        # Add original image, processed image, final image and diseased percentage to PDF
        elements.append(Paragraph("\n", normal_style)) 
        elements.append(Paragraph(f"Original Image: {image_file}", normal_style))
        elements.append(Image(image_path, width=400, height=300))
        elements.append(Paragraph("\n", normal_style)) 
        #elements.append(Paragraph("Overlay Mask:", normal_style))
        #elements.append(Image(img_data, width=400, height=300))
        elements.append(Paragraph("\n", normal_style)) 
        elements.append(Paragraph("Diseased Part:", normal_style))  
        elements.append(Image(final_img_data, width=400, height=300))
        elements.append(Paragraph("\n", normal_style)) 
        elements.append(Paragraph(diseased_text, normal_style))
        elements.append(Paragraph("\n", normal_style)) 
        elements.append(Paragraph("\n", normal_style)) 
        elements.append(Paragraph("\n", normal_style)) 

    # Build the PDF document
    doc.build(elements)
    print("PDF has been generated, please check output folder!")
        
# input_folder = r"C:\Users\abdul\Desktop\Foliar research\Cleaned Sample Data\Tomato_Early_blight\5b86ab6a-3823-4886-85fd-02190898563c___RS_Erly.B 8452_no_bg.png"
# output_destination = r"C:\Users\abdul\Desktop"  # Output destination for the PDF
input_folder_path = filedialog.askdirectory(title='Select folder with images to be processed in PDF')
output_folder_path = filedialog.askdirectory(title='Select output folder to save PDF file')

brown_edge_detection(input_folder_path, output_folder_path)