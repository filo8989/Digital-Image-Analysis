import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, measure
import numpy as np
import cv2
from PIL import Image

# Increase PIL limit for image size
Image.MAX_IMAGE_PIXELS = None

def generate_segment_color_image(image, labeled_image):
    segment_color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for region in measure.regionprops(labeled_image):
        segment_label = region.label
        color = np.random.randint(0, 255, 3)  # Generate a random color for the segment
        
        # Find the coordinates of the pixels corresponding to the segment
        y_coords, x_coords = np.where(labeled_image == segment_label)

        # Assign the color to the corresponding pixels
        segment_color_image[y_coords, x_coords] = color
        
        # Add the segment number to the overlay
        centroid_y, centroid_x = map(int, region.centroid)
        cv2.putText(segment_color_image, str(segment_label), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return segment_color_image

def analyze_images_in_folder(input_folder_path, output_folder_path):
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(input_folder_path, image_file)
        
        image = io.imread(image_path)

        # Convert to grayscale if necessary
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        # Threshold
        threshold = 100  # You may need to adjust this threshold value
        binary_image = image > threshold

        labeled_image = measure.label(binary_image)
        segment_color_image = generate_segment_color_image(image, labeled_image)

        # Display and save colored image
        plt.figure(figsize=(10, 5), dpi=700)
        plt.imshow(segment_color_image)
        plt.axis('off')
        
        # Save colored image
        colored_image_path = os.path.join(output_folder_path, f"{os.path.splitext(image_file)[0]}_colored.png")
        plt.savefig(colored_image_path, bbox_inches='tight', pad_inches=0.05, dpi=700)
        plt.close()

        print(f"Colored image saved as: {colored_image_path}")

        # Compute measurements
        properties = measure.regionprops(labeled_image)

        # Create DataFrame
        results = []
        for idx, prop in enumerate(properties, start=1):
            # Get area
            area = prop.area

            # Get major axis length
            axis_major_length = prop.major_axis_length

            # Get minor axis length
            axis_minor_length = prop.minor_axis_length

            # Get eccentricity
            eccentricity = prop.eccentricity

            # Get orientation (angle in radians)
            orientation = np.degrees(prop.orientation)

            results.append({
                "Segment": idx,
                "Area": area,
                "Major Axis Length": axis_major_length,
                "Minor Axis Length": axis_minor_length,
                "Eccentricity": eccentricity,
                "Orientation (degrees)": orientation
            })

        df = pd.DataFrame(results)

        # Generate measurement table as a PDF
        table_pdf_path = os.path.join(output_folder_path, f"{os.path.splitext(image_file)[0]}_measurement_table.pdf")
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize as needed
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center')

        # Save measurement table as PDF
        plt.savefig(table_pdf_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()

        print(f"Measurement table saved as: {table_pdf_path}")

input_folder_path = "/Users/kirolololololos/Desktop/V1 Release/Measurements_Slide"
output_folder_path = "/Users/kirolololololos/Desktop/V1 Release/Measurements_Slide"
analyze_images_in_folder(input_folder_path, output_folder_path)
