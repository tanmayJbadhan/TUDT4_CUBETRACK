import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_cube_faces(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # Convert to HSV color space to detect blue color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            # Extract the cube face
            cube_face = img[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cube_face)

image_folder = 'CubeFace_Artist'
output_folder = 'CubeFace_Extracted'
extract_cube_faces(image_folder, output_folder)
