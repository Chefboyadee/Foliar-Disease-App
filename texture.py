import cv2
import numpy as np

def extract_disease_area(image_path, ksize=5, sigma=1.0, theta=0, lambd=1.0, gamma=0.02, canny_low_threshold=50, canny_high_threshold=150, threshold_value=100, dilation_kernel_size=3, erosion_kernel_size=3):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filter
    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    gabor_result = cv2.filter2D(gray_img, cv2.CV_64F, gabor_filter)

    # Normalize the result to bring values in the range [0, 255]
    normalized_result = cv2.normalize(gabor_result, None, 0, 255, cv2.NORM_MINMAX)

    # Convert Gabor result to uint8 for Canny edge detection
    gabor_uint8 = normalized_result.astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(gabor_uint8, canny_low_threshold, canny_high_threshold)

    # Thresholding to extract disease areas
    _, binary_mask = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological operations (dilation and erosion)
    kernel_dilation = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel_dilation, iterations=1)

    kernel_erosion = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel_erosion, iterations=1)

    color_binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Display the result in a window
    cv2.imshow("orginal", img)
    cv2.imshow('Disease Areas', binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Images/No_BG/septoria_leaf/0a68a294-30d1-4422-ab7e-a1909ec277f7___JR_Sept.L.S 8443_no_bg.png'
extract_disease_area(image_path)
