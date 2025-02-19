from PIL import Image
import os

def combine_images(images_path, output_folder):
    """
    Combines a series of PNG images into a single image arranged vertically with 16 images per column.

    Args:
        images_path (str): Path to the directory containing input images.
        output_folder (str): Path to the output folder where the combined image will be saved.

    Returns:
        None
    """
    # List PNG files in the images folder
    png_list = [file for file in os.listdir(images_path) if file.lower().endswith('.png')]
    png_list_sorted = sorted(png_list)  # Sort the list of image filenames

    images = [Image.open(os.path.join(images_path, image)) for image in png_list_sorted]

    num_images = len(images)
    if num_images == 0:
        print("No images found in the specified folder.")
        return

    # Calculate the number of columns required
    num_columns = (num_images + 15) // 16  # Ceiling division to ensure at least num_images columns

    # Calculate the maximum width and height of the combined image
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a blank canvas for the combined image
    combined_image_width = num_columns * max_width
    combined_image_height = 16 * max_height
    combined_image = Image.new('RGB', (combined_image_width, combined_image_height), (255, 255, 255))

    # Paste images onto the canvas
    x_offset = 0
    y_offset = 0
    count = 0
    for img in images:
        combined_image.paste(img, (x_offset, y_offset))
        y_offset += max_height
        count += 1
        if count == 16:
            count = 0
            y_offset = 0
            x_offset += max_width

    # Save and display the combined image
    output_path = os.path.join(output_folder, 'Detected_Slide.png')
    combined_image.save(output_path)
    print(f"Combined image saved at: {output_path}")
    combined_image.show()

if __name__ == "__main__":
    images_path = r'/Users/kirolololololos/Desktop/V1 Release/Detected_Tiles'
    output_folder = '/Users/kirolololololos/Desktop/V1 Release/Segmented_Slide'
    combine_images(images_path, output_folder)
