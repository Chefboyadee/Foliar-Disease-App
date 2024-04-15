import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib

def preprocess_and_extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological transform to close small holes
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Perform bitwise AND with original image to get segmented leaf
    segmented = cv2.bitwise_and(image, image, mask=closed)

    # Extract shape features
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = cv2.contourArea(contours[0])
    leaf_perimeter = cv2.arcLength(contours[0], True)

    # Extract color features
    b, g, r = cv2.split(segmented)
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_std = np.std(b)
    g_std = np.std(g)
    r_std = np.std(r)

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

    # Assemble the feature vector
    #features = [leaf_area, leaf_perimeter] + color_mean + color_std + [green_ratio, non_green_ratio, contrast, dissimilarity, homogeneity, energy, correlation]
    features = [leaf_area, leaf_perimeter + b_mean, g_mean, r_mean, b_std, g_std, r_std + contrast, dissimilarity]
    
    return np.array(features)

# Load the saved CatBoost model
catboost_model_path = r"C:\Users\abdul\Desktop\Foliar research\Random Forest Model\random_forest_model.joblib"
catboost_model = joblib.load(catboost_model_path)

# Example usage
features = preprocess_and_extract_features(r"C:\Users\abdul\Desktop\Foliar research\plantvillage dataset\segmented\Tomato___Tomato_Yellow_Leaf_Curl_Virus\2cc6d499-bf92-48c6-b0f9-6d9ef3c2f76e___UF.GRC_YLCV_Lab 03271_final_masked.jpg")
print(features)


# Convert features to numpy array and reshape
features = np.array(features).reshape(-1, 1)
# Predict using the loaded model
prediction = catboost_model.predict(features)
print("Prediction:", prediction)
