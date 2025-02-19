
"""
@author: OMER
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import measure
import pandas as pd
import os

# Define the Path
PATH = "D:/Ribulin_Tiff_Files"

# Read all folders in the PATH
folders = [item for item in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, item))]

# Define the props
# see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

all_props = ["area", "axis_major_length", "axis_minor_length", "eccentricity", "orientation"]

# Parameters for the segmentation

iterations_for_dilate = 4
kernel_for_structuring_element_for_segmentation = (15, 15)
kernel_for_structuring_element_for_closing = (15, 15)
iterations_for_closing = 3

# SEGMENTATION
for folder in folders:
    # Read the image
    general_path = os.path.join(PATH, folder)
    image_path = general_path + "/" + folder + ".jpg"
    original = cv2.imread(image_path)

    print(f"Image - {image_path} is read")

    # Create the directory IMAGE_NUMBER_SEGMENTATION
    if not os.path.exists(f"{general_path}/{folder}_SEGMENTATION"):
        os.makedirs(f"{general_path}/{folder}_SEGMENTATION")

        # Directory to save
    directory_to_save = general_path + "/" + folder + "_SEGMENTATION/"

    # Vessels are best detected in HSV space. Especially in H channel
    # Convert RGB image to HSV and take H channel

    h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(h_s_v_image)
    h = 255 - h

    ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
    binary = binary.astype(np.uint8)
    binary *= 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_for_structuring_element_for_segmentation)
    binary = cv2.dilate(binary, kernel, iterations=iterations_for_dilate)

    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(binary, [c], 0, (255, 255, 255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_for_structuring_element_for_closing)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations_for_closing)

    # Segmentation is done, saved it into the folder that was created before

    plt.imsave(f"{directory_to_save}{folder}_Segmented.png", opening, cmap="gray")

    # Calculate the props
    label_img = label(opening)
    props = regionprops(label_img)

    props = measure.regionprops_table(label_img, original, properties=all_props)

    # Do not know if this line is necessary or not
    props["area"] = props["area"] * 0.25 ** 2
    # Do not know if this line is necessary or not

    data = pd.DataFrame(props)

    # Save the propos
    data.to_csv(f'{directory_to_save}Measurements_{folder}.csv', index=True)

    # Show the detected vessels on original image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        cv2.drawContours(original, [c], 0, (0, 255, 0), 5)

    # Save detected vessels        
    cv2.imwrite(f"{directory_to_save}{folder}_Vessels.jpg", original)
