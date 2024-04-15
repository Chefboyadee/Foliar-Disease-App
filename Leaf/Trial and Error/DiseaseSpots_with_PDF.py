import cv2
import numpy as np
import os
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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

        # Calculate the total number of pixels
        total_pixels = original_image.shape[0] * original_image.shape[1]

        # Calculate the percentage of unhealthy pixels
        unhealthy_pixels = np.count_nonzero(brown_mask | green_mask | yellow_mask)
        unhealthy_percentage = (unhealthy_pixels / total_pixels) * 100

        # Calculate the percentage of healthy pixels
        healthy_percentage = 100 - unhealthy_percentage

        # Create an empty mask for the overlay
        overlay_mask = np.zeros_like(original_image)

        # Set the brown, green, and yellow regions in the overlay mask
        overlay_mask[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to red
        overlay_mask[green_mask != 0] = [0, 255, 0]  # Set green pixels to green
        overlay_mask[yellow_mask != 0] = [0, 255, 255]  # Set yellow pixels to yellow
# Create a text content for healthy percentage
        healthy_text = f"The percentage of healthy features (pixels) on this leaf is: {healthy_percentage:.2f}%"

        # Overlay the mask on the original image
        processed_image = cv2.addWeighted(original_image, 0.5, overlay_mask, 0.5, 0)

        
        # Encode processed image to PNG format
        ret, buffer = cv2.imencode('.png', processed_image)
        if not ret:
            print("Error: Unable to encode processed image.")
            continue

        # Create in-memory file-like object
        img_data = io.BytesIO(buffer)

        # Add original image, processed image, and healthy percentage to PDF
        elements.append(Paragraph(f"Original Image: {image_file}", normal_style))
        elements.append(Image(image_path, width=400, height=300))
        elements.append(Paragraph("Processed Image:", normal_style))
        elements.append(Image(img_data, width=400, height=300))
        elements.append(Paragraph(healthy_text, normal_style))
        elements.append(Paragraph("\n", normal_style))  # Add some space between images

    # Build the PDF document
    doc.build(elements)

input_folder = r"C:\Users\abdul\Desktop\Foliar research\bruh"  # Input folder containing images
output_destination = r"C:\Users\abdul\Desktop\Foliar research\bruh"  # Output destination for the PDF
brown_edge_detection(input_folder, output_destination)

