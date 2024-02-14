import cv2
import numpy as np

def brown_edge_detection(image_path):
    # Read the image
    original_image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the brown color in HSV
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([20, 255, 200])

    lower_green = np.array([40, 60, 20])
    upper_green = np.array([70, 255, 200])

    lower_yellow = np.array([25, 60, 20])
    upper_yellow = np.array([39, 255, 200])

    # Create a mask for the brown color
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    green_mask = cv2.inRange(hsv_image,lower_green,upper_green)

    yellow_mask = cv2.inRange(hsv_image,lower_yellow,upper_yellow)

    # Calculate the percentage of brown and not brown pixels
    total_pixels = original_image.shape[0] * original_image.shape[1]
    unhealthy_percent = (np.count_nonzero(brown_mask+yellow_mask) / total_pixels) * 100
    healthy_percent = 100 - unhealthy_percent


    print(f"Percentage of unhealthy pixels: {unhealthy_percent:.2f}%")
    print(f"Percentage of healthy pixels: {healthy_percent:.2f}%")

    

    # Create an image with the original color where brown pixels are detected
    brown_color_image = original_image.copy()
    brown_color_image[brown_mask != 0] = [0, 0, 255]  # Set brown pixels to green for visualization

   # Create an image with the original color where brown pixels are detected
    green_color_image = original_image.copy()
    green_color_image[green_mask != 0] = [0, 255, 0]  # Set brown pixels to green for visualization

    # Create an image with the original color where brown pixels are detected
    yellow_color_image = original_image.copy()
    yellow_color_image[yellow_mask != 0] = [0, 255, 255]  # Set brown pixels to green for visualization

    

    # Display the original image, the detected edges, the brown mask, and the result
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Brown Color Image', brown_color_image)
    cv2.imshow('green image', green_color_image)
    cv2.imshow('yellow image', yellow_color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/late_blight/0ab172fa-ac7b-4a3e-90c8-0708703eb3bb___RS_Late.B 5572_no_bg.png'
brown_edge_detection(image_path)