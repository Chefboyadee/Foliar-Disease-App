import numpy as np
import cv2

image = cv2.imread('Images/TestDataset/4f1ea8d7-f41e-4d33-9e01-4ea75ef249e7___RS_HL_0737_no_bg.png')
blank_mask = np.zeros(image.shape, dtype=np.uint8)
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 44])
upper = np.array([72, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    cv2.drawContours(blank_mask,[c], -1, (255,255,255), -1)
    break

result = cv2.bitwise_and(original,blank_mask)

cv2.imshow('Original', original)
cv2.imshow('result', result)
cv2.waitKey()