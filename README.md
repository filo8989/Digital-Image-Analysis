# Vessel Analysis 

## Overview
The following pipeline offers a robust, reproducible, and scalable solution for extracting vessel-related features from histological images. Accurate vascular network segmentation provides key insights into vascular conformation and remodeling. While particularly relevant in oncology for characterizing the tumor microenvironment, it may also be applied to other diseases where vascular alterations are of paramount importance, including those involving inflammation, fibrosis, and microvascular dysfunction. By enhancing our understanding of vascular-mediated disease progression and treatment response, this approach supports both research and clinical decision-making.

## Objectives
- **Quantitative Analysis**: Extracts precise vessel morphology and spatial distribution metrics.
- **Automated & Scalable**: Processes large datasets efficiently, reducing manual workload.
- **Open-Source & Modifiable**: Easily adaptable to different research contexts and imaging modalities.
- **Interdisciplinary Integration**: Combines pathology and bioinformatic image analysis.

## Getting Started
1. **Prepare Your Data**:
   - 1.1 Stain the histological images of interest for CD31
   - 1.2 Ensure the images are formatted correctly (e.g., `.tiff`, `.jpg`)
2. **Visualize the Slide in QuPath & Perform Tiling**:
   - 2.1 Image visualization and annotation
   - 2.2 Tile creation
3. **Segment Vessels and Extract Features in Python**:
   - 3.1 Convertion of RGB to HSV format
   - 3.2 Application of Otsu’s thresholding
   - 3.3 Tile stitching
   - 3.4 Morphological operations
   - 3.5 Feature extraction
4. **Interpret Your Data**

---

## Step-by-Step Guide
So, you've some histological images and a good dose of scientific curiosity. Now what? Let's walk through the pipeline together, breaking it down step by step.

### 1. **Prepare Your Data**
Before diving into analysis, we need to prepare your images. Here’s what you need to get your data ready for processing:

   - #### 1.1 Stain the histological images of interest for CD31
   The first step is to obtain the histological images of interest. These images must include a stain that highlights blood vessels. Common choices are CD31 or CD34, both of which label the endothelial cells lining the vasculature. CD31 is particularly useful due to its strong and specific endothelial staining, while CD34 provides additional coverage of endothelial and progenitor cells. These stains are essential for enhancing vessel contrast and enabling analyses further down the pipeline. Finally, the slides need to be scanned digitally for further evaluation.

   - #### 1.2 Ensure the images are formatted correctly (e.g., `.tiff`, `.jpg`)
     Once you have your images, it’s crucial that they are in the right format for compatibility with the analysis pipeline. The most commonly used image formats for this type of work are .tiff (higher quality, slower processing) and .jpg (lower quality, faster processing).

### 2. **Visualize the Slide in QuPath & Perform Tiling**

 - #### 2.1 Image visualization and annotation
   Before we can analyze vessels, we need to load the slides into QuPath, a powerful image analysis software for digital pathology. Consider it your digital microscope where you can zoom, pan, and annotate. Use QuPath’s built-in tools to annotate regions of interest (ROIs) containing the vascular structures you wish to analyse. You may also adjust visualization settings to enhance contrast and highlight vascular features for better segmentation.
   
 - #### 2.2 Tile creation
   Now, break down your images into digestible tiles. Imagine you're a cartographer mapping out a new land. You wouldn’t study the entire continent at once, right? You’d divide it into sections. That’s exactly why tiling is a crucial preprocessing step for users with images that are too large to process efficiently all at once, so they are divided into smaller, more manageable tiles. This is how vessels are captured with adequate detail while maintaining computational efficiency. 
    - Navigate to: "Analyze → Tiles & Superpixels → Create Tiles" to generate sub-images, or tiles.
    - Set tile size based on the expected vessel dimensions, ensuring adequate resolution for segmentation.
    - Execute the "ExportingImages.groovy" script as follows:

      Firstly, preliminary setup requires importation of the necessary QuPath libraries for image writing and region requests, enabling you to manipulate images and regions within QuPath.
      ```groovy
      import qupath.lib.images.writers.ImageWriterTools
      import qupath.lib.regions.RegionRequest
      import qupath.lib.gui.scripting.QPEx
      ```
      Here, you define the path where the tiles will be saved, you retrieve the current image data and server, and you retrieve the hierarchy of the image data and the annotations present in the image. This information is essential for accessing the images you will process.
      ```groovy
      def path = "Insert/Path/to/Slide"
      
      def imageData = getCurrentImageData()
      def server = getCurrentServer()
      def filename = ""
      
      def hierarchy = imageData.getHierarchy()
      def annotations = hierarchy.getAnnotationObjects()
      i = 1
      ```
      In this loop, the script processes each annotation. It retrieves the ROI and creates a region request for that ROI. If the annotation type is "Image," it generates a tile name and saves the region as a .jpg file in the specified path. This process automates tile extraction and ensures that the relevant details are preserved.
      ```groovy
      for (annotation in annotations) {
          roi = annotation.getROI()
          
          print(annotation.getName())
          
          def request = RegionRequest.createInstance(imageData.getServerPath(), 2, roi)
          
          tiletype = annotation.getParent().getPathClass()
          print(tiletype.toString())
          
          if (tiletype.toString().equals("Image")) {
          
              String tilename = String.format("InsertSlideName.jpg")
          
              ImageWriterTools.writeImageRegion(server, request, path + "/" + tilename)
          }
      }
      ```
   - After running the entire script, check the tile output for completeness and quality to ensure that all relevant details are preserved before proceeding to segmentation. Are you all set? Let’s proceed to the next section.

### 3. **Segment Vessels and Extract Features in Python**
This is where the true transformation — or as some may say, magic— happens. With the image tiles ready, you can perform vessel segmentation and feature extraction using Python. Simply run the "Segmentation.py" script as follows.

In the prelinimary setup, upload necessary libraries.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import measure
import pandas as pd
import os
```

Next, define the path that sets the directory where the images are stored and create a list of subdirectories (folders) within the specified path.
```python
PATH = "Insert/Path/to/Folders"

folders = [item for item in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, item))]
```

Now, loop through each folder to process the images contained within, construct the image path, and read the image using OpenCV. Additionally, you can check if a directory for saving segmented images exists, and create it if not.
```python
for folder in folders:
    general_path = os.path.join(PATH, folder)
    image_path = general_path + "/" + folder + ".jpg"
    original = cv2.imread(image_path)

    print(f"Image - {image_path} is read")

    if not os.path.exists(f"{general_path}/{folder}_SEGMENTATION"):
        os.makedirs(f"{general_path}/{folder}_SEGMENTATION")

    directory_to_save = general_path + "/" + folder + "_SEGMENTATION/"  
```

- #### 3.1 Convertion of RGB to HSV format
  Blood vessels love to hide, but their secrets can be revealed with the right tips and tricks. The first trick is converting the extracted tiles from the standard RGB (Red, Green, Blue) colour space to HSV (Hue, Saturation, Value) format. This transformation is particularly useful because the H channel enhances the contrast of blood vessels, making them easier to distinguish from the surrounding tissue. Using the OpenCV library, the conversion is straightforward
   ```python
   h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(h_s_v_image)
   h = 255 - h
   ```
   
- #### 3.2 Application of Otsu’s thresholding
  Once the H channel is extracted, Otsu’s thresholding is applied to create a binary mask. This technique automatically determines an optimal threshold, separating vessel structures from the background. 
  ```python
  ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
  binary = binary.astype(np.uint8)
  binary *= 255
  ```
  
- #### 3.3 Tile stitching
  For now, you're left with individual segmented tiles. However, like solving a puzzle where you’ve analyzed each piece individually, it’s time to put them back together to see the bigger picture.
  ??????
  
- #### 3.4 Morphological operations
  You fine-tune the segmentation by cleaning up noise and ensuring vessel structures are continuous thanks to morphological operations. 

  Feel free to play with the parameters below to tailor the output to your necessities, according to the characteristics of your histological images and the level of detail required for your analysis.
  `kernel_for_structuring_element_for_segmentation` defines the size of the structuring element used during segmentation. Increasing its size results in a broader effect, which can help connect fragmented vessels but may also lead to over-segmentation. Reducing it preserves finer details but might leave gaps in vessel structures.
`kernel_for_structuring_element_for_closing` determines the size of the kernel used for morphological closing (dilation followed by erosion), which helps fill small gaps in vessel segmentation. A larger kernel merges nearby structures more aggressively, while a smaller one maintains finer vessel details.
`iterations_for_dilate` controls how many times the dilation operation is applied. More iterations expand the segmented vessel regions, potentially merging adjacent vessels, while fewer iterations maintain a more conservative segmentation.
`iterations_for_closing` specifies the number of times closing is performed. Higher values result in more pronounced gap-filling and smoother vessel structures, whereas lower values retain more original segmentation details.
   ```python
   kernel_for_structuring_element_for_segmentation = (15, 15)
   kernel_for_structuring_element_for_closing = (15, 15)
   iterations_for_dilate = 4
   iterations_for_closing = 3
    ```

   Dilation is used to bridge small gaps between fragmented vessel structures, improving connectivity. Closing, which involves dilation followed by erosion, eliminates small holes within vessels. Additionally, you may find the boundaries of the detected vessels to fill them in the binary image.
   ```python
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_for_structuring_element_for_segmentation)
   binary = cv2.dilate(binary, kernel, iterations=iterations_for_dilate)

   cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] if len(cnts) == 2 else cnts[1]

   for c in cnts:
      cv2.drawContours(binary, [c], 0, (255, 255, 255), -1)
   
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_for_structuring_element_for_closing)
   opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations_for_closing)
   ```

   Now that you have finalized the segmentation, save the processed image to the specified directory.
   ```python
   plt.imsave(f"{directory_to_save}{folder}_Segmented.png", opening, cmap="gray")
   ```
  Example output: a binary image, created using a mask, where all vessels are represented in white and the background is shown in black.

  <img src="/Segmented.png" alt="Binary image output after segmentation and morphological operations" width="300"/>
  
   To enhance interpretability - because seeing is believing - contour detection outlines vessel boundaries as an overlay on the original images. Nothing beats a well-labeled, highlighted image that speaks for itself, showcasing exactly what you've extracted.
   ```python
   contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
   for c in contours:
      cv2.drawContours(original, [c], 0, (0, 255, 0), 5)
       
    cv2.imwrite(f"{directory_to_save}{folder}_Vessels.jpg", original)
   ```
   Example output: the original histological image, with vessel countours highlighted in green.
  
   <img src="/Contoured.jpg" alt="Histological image with vessel contours overlayed" width="300"/>
    
- #### 3.5 Feature extraction
  Great, you now have segmented vessels, but science demands numbers, not just pretty pictures. To obtain more information on vessel morphology, quantitative features are extracted using `regionprops_table` from `skimage.measure`. These properties include:
   - `area`: The number of pixels inside each segmented vessel.
   - `axis_major_length`: The length of the longest axis of the vessel (major axis of the fitted ellipse).
   - `axis_minor_length`: The length of the shortest axis of the vessel (minor axis of the fitted ellipse).
   - `eccentricity`: Measures how close to a circle the vessel is (0 = perfect circle, between 0 and 1 = ellipse, 1 = parabola).
   - `orientation`: The angle in degrees of the major axis relative to the horizontal axis.

  The `label()` function from `skimage.measure` assigns a unique integer label to each connected component - in our case a segmented vessel - in the binary image, this helps in identifying individual vessels. You then define the list of desired morphological properties and compute them.
  ```python
  label_img = label(opening)
  all_props = ["area", "axis_major_length", "axis_minor_length", "eccentricity", "orientation"]
  props = regionprops(label_img)
  props = measure.regionprops_table(label_img, original, properties=all_props)
  data = pd.DataFrame(props)
  data.to_csv(f'{directory_to_save}Measurements_{folder}.csv', index=True)
  ```
   The output is a CSV file `{folder}_Measurements.csv` containing quantitative measurements of each detected vessel in the segmented image. The columns of the CSV file represent different morphological properties of the vessels, and each row corresponds to one labeled vessel.

### 4. **Interpret Your Data**
Finally, you have arrived at the last step. This is where the numbers should start to talk, where images should transform into knowledge, and where you should ask yourself, **“So what?”**, what do these findings mean for your future research? And just like that, you’ve made sense of it all, you've taken a raw histological image and extracted meaningful biological insights. So, go forth and analyze, because in the world of vessel analysis, the smallest capillary could hold the biggest discovery.

## Contributors 
G. Grion, K. Roufail, F. E. Colella, Dr. Ö. Mintemur, Dr. S. L. Renne.

## Citation
If you use this pipeline, please cite:
> Renne et al., Integrating Spatial Vessel Analysis into Tumour Microenvironment Characterization, *Virchows Arch*, 485(Suppl 1), 2024.

## Contact
For inquiries, please reach out via [GitHub Issues](https://github.com/filo8989/Digital-Image-Analysis/issues).
