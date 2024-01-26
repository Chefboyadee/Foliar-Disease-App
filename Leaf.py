import numpy as np 
import cv2
import os


folder_path = 'Images/Test'
output_folder_path = 'Images/Cleaned'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
       
        
        
      # load image
        img =  cv2.imread(file_path)
       
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # find the green color 
        mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
        # find the brown color
        mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
        # find the yellow color in the leaf
        mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

        # find any of the three colors(green or brown or yellow) in the image
        mask = cv2.bitwise_or(mask_green, mask_brown)
        mask = cv2.bitwise_or(mask, mask_yellow)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
            output_path = os.path.join(output_folder_path,filename)
            cv2.imwrite(output_path,res)
        else:
            output_path = os.path.join(output_folder_path,filename)
            cv2.imwrite(output_path,res)
        

        
       