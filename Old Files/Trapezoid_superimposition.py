from PIL import Image
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

Image.MAX_IMAGE_PIXELS = None

def convert_image_to_tiff(input_path, output_folder, max_size=(10000, 10000)):
    img = Image.open(input_path)
    img.thumbnail(max_size, Image.LANCZOS)  # Resize image if it's too large
    tiff_path = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.tiff')
    img.save(tiff_path, 'TIFF')
    return tiff_path

def resize_image_with_opencv(input_path, target_size):
    img = cv2.imread(input_path)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

def compute_ssim_with_skimage(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_index, _ = ssim(img1_gray, img2_gray, full=True)
    print(f"SSIM Index: {ssim_index}")
    return ssim_index

def align_images_by_points(img1, img2, points_img1, points_img2):
    h, status = cv2.findHomography(np.array(points_img2), np.array(points_img1))
    height, width, channels = img1.shape
    img2_aligned = cv2.warpPerspective(img2, h, (width, height))
    return img2_aligned

def main(thresholded_png_path, maldi_tiff_path, output_folder):
    if not os.path.isfile(thresholded_png_path):
        print(f"Error: {thresholded_png_path} not found.")
        return
    if not os.path.isfile(maldi_tiff_path):
        print(f"Error: {maldi_tiff_path} not found.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert images to TIFF with resizing
    thresholded_tiff_path = convert_image_to_tiff(thresholded_png_path, output_folder)
    maldi_tiff_path = convert_image_to_tiff(maldi_tiff_path, output_folder)
    
    # Read converted TIFF images
    thresholded_img = cv2.imread(thresholded_tiff_path)
    maldi_img = cv2.imread(maldi_tiff_path)
    
    # Resize MALDI image
    resized_maldi_img = resize_image_with_opencv(maldi_tiff_path, (thresholded_img.shape[1], thresholded_img.shape[0]))
    
    # Compute SSIM
    compute_ssim_with_skimage(thresholded_img, resized_maldi_img)
    
    # Points for alignment (example points, adjust as needed)
    points_thresholded = [[100, 100], [300, 100], [100, 300], [300, 300]]
    points_maldi = [[50, 50], [200, 50], [50, 200], [200, 200]]
    
    # Align images
    aligned_maldi_img = align_images_by_points(thresholded_img, resized_maldi_img, points_thresholded, points_maldi)
    
    # Superimpose images
    superimposed_img = cv2.addWeighted(thresholded_img, 0.5, aligned_maldi_img, 0.5, 0)
    
    # Save and show the superimposed image
    superimposed_tiff_path = os.path.join(output_folder, 'superimposed.tiff')
    cv2.imwrite(superimposed_tiff_path, superimposed_img)
    
    superimposed_img_pil = Image.open(superimposed_tiff_path)
    superimposed_img_pil.show()

# Example paths (make sure these are correct)
thresholded_png_path = '/Users/kirolololololos/Desktop/cartella senza nome/thresholded_image.png'
maldi_tiff_path = '/Users/kirolololololos/Desktop/cartella senza nome 2/maldi_image.tiff'
output_folder = '/Users/kirolololololos/Documents/Scrivania Stuff/V1 Release/Segmented_Slide'

main(thresholded_png_path, maldi_tiff_path, output_folder)
