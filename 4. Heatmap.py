import cv2
import os

# Load images
img_path = '/Users/kirolololololos/Desktop/Digital Image Analysis/Alpha Version/Segmented_Slide/Detected_Slide.png'
bin_img_path = '/Users/kirolololololos/Desktop/Digital Image Analysis/Alpha Version/Segmented_Slide/Thresholded_Slide.png'

img = cv2.imread(img_path)
bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if img is None or bin_img is None:
    print("Error: Failed to load images.")
    exit()

# Apply colormap to the binary image
heatmap_img = cv2.applyColorMap(bin_img, cv2.COLORMAP_JET)

# Set blue channel to zero to remove the blue part
heatmap_img[:, :, 0] = 0

# Gaussian blur
blur = cv2.GaussianBlur(heatmap_img, (35, 35), 0)

# Overlay heatmap on original image with adjusted weights
super_imposed_img = cv2.addWeighted(blur, 0.5, img, 0.5, 0)

# Specify the folder path where you want to save the image
output_folder = '/Users/kirolololololos/Desktop/Digital Image Analysis/Alpha Version/Segmented_Slide'

# Ensure the output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the resulting image in the output folder
output_path = os.path.join(output_folder, 'Hitmap_Slide.png')
cv2.imwrite(output_path, super_imposed_img)

print("Image saved successfully at:", output_path)
