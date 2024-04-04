import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def contour_find(input_folder, output_folder_detected, output_folder_thresholded):
    if not os.path.exists(output_folder_detected):
        os.makedirs(output_folder_detected)
    if not os.path.exists(output_folder_thresholded):
        os.makedirs(output_folder_thresholded)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            # Construct the full path to the image
            image_path = os.path.join(input_folder, filename)

            # Perform contour detection on the current image
            process_single_image(image_path, output_folder_detected, output_folder_thresholded, filename)

def process_single_image(image_path, output_folder_detected, output_folder_thresholded, filename):
    original = Image.open(image_path)
    original_ = np.array(original)
    original = np.array(original.convert('HSV'))
    channel = 0
    
    original = cv2.medianBlur(original, 3)
    
    binary = cv2.Canny(original[:, :, channel], 150, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    binary = cv2.dilate(binary, kernel, iterations=10)

    contours, hierarchy = cv2.findContours(image=binary, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas with the same dimensions as the original image
    canvas = np.zeros_like(original_)

    # Draw contours on the blank canvas
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 7)

    # Overlay the canvas onto the original image
    result = cv2.addWeighted(original_, 1, canvas, 1, 0)

    # Save the processed image with contours to output_folder_detected
    output_detected_path = os.path.join(output_folder_detected, f'Detected_{os.path.splitext(os.path.basename(image_path))[0]}.png')
    cv2.imwrite(output_detected_path, result)

    # Save the thresholded image to output_folder_thresholded
    thresholded_output_path = os.path.join(output_folder_thresholded, f'Thresholded_{os.path.splitext(filename)[0]}.png')
    cv2.imwrite(thresholded_output_path, binary)


# Example usage:
input_folder = '/Users/kirolololololos/Desktop/V1 Release/Tiles'
output_folder_detected = '/Users/kirolololololos/Desktop/V1 Release/Detected_Tiles'
output_folder_thresholded = '/Users/kirolololololos/Desktop/V1 Release/Thresholded_Tiles'

contour_find(input_folder, output_folder_detected, output_folder_thresholded)
