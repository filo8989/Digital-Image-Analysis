from PIL import Image, ImageChops, ImageOps, ImageStat
import os

# Increase the maximum allowable image size (be cautious with this)
Image.MAX_IMAGE_PIXELS = None

def convert_image_to_tiff(input_path, output_folder):
    img = Image.open(input_path)
    
    # Convert image to grayscale if it's not already in grayscale mode
    if img.mode in ('RGBA', 'RGB'):
        img = img.convert('L')
    
    tiff_path = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.tiff')
    img.save(tiff_path, 'TIFF')
    return tiff_path

def get_image_properties(image_path):
    img = Image.open(image_path)
    return img.size, img.info.get('dpi', (72, 72))

def resize_image(input_path, output_path, target_size):
    img = Image.open(input_path)
    img = img.resize(target_size, Image.LANCZOS)
    img.save(output_path, 'TIFF')
    return output_path

def compute_similarity_and_superimpose(img1_path, img2_path, output_path):
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    # Compute similarity using basic approach (normalized difference)
    diff = ImageChops.difference(img1, img2)
    diff = ImageOps.autocontrast(diff)  # Enhance the contrast of the difference
    similarity_score = ImageStat.Stat(diff).mean[0]  # Mean difference value

    print(f"Similarity score: {similarity_score}")

    # Superimpose images
    img1 = Image.open(img1_path).convert('RGBA')
    img2 = Image.open(img2_path).convert('RGBA')
    superimposed_img = Image.blend(img1, img2, alpha=0.5)

    superimposed_img.save(output_path, 'TIFF')
    return output_path

def main(thresholded_png_path, maldi_png_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Convert images to TIFF
    thresholded_tiff_path = convert_image_to_tiff(thresholded_png_path, output_folder)
    maldi_tiff_path = convert_image_to_tiff(maldi_png_path, output_folder)

    # Step 2: Check dimensions and resolution
    thresholded_size, thresholded_dpi = get_image_properties(thresholded_tiff_path)
    maldi_size, maldi_dpi = get_image_properties(maldi_tiff_path)

    print(f"Thresholded Image Size: {thresholded_size}, DPI: {thresholded_dpi}")
    print(f"MALDI Image Size: {maldi_size}, DPI: {maldi_dpi}")

    # Step 3: Resize MALDI TIFF to match the thresholded TIFF
    resized_maldi_tiff_path = os.path.join(output_folder, 'resized_maldi.tiff')
    resize_image(maldi_tiff_path, resized_maldi_tiff_path, thresholded_size)

    # Step 4: Compute similarity and superimpose
    superimposed_tiff_path = os.path.join(output_folder, 'superimposed.tiff')
    compute_similarity_and_superimpose(thresholded_tiff_path, resized_maldi_tiff_path, superimposed_tiff_path)

    # Display the superimposed image
    superimposed_img = Image.open(superimposed_tiff_path)
    superimposed_img.show()

# Example usage
thresholded_png_path = '/Users/filippocolella/Desktop/Digital Image Analysis/Full_Image_Closed.png'
maldi_png_path = '/Users/filippocolella/Desktop/Digital Image Analysis/oleic.tiff'
output_folder = '/Users/filippocolella/Desktop/Digital Image Analysis/superimposed and tiff'

main(thresholded_png_path, maldi_png_path, output_folder)
