# Vessel Analysis

## Overview
Vascular networks play a critical role in tumour progression, therapeutic response, and disease prognosis. Accurate vessel segmentation and spatial analysis provide key insights into how tumours evolve and how drugs penetrate different tissue regions. The **Erivessel** pipeline offers a robust, reproducible, and scalable solution for extracting vessel-related features from histological images. By integrating spatial vessel analysis with tumour microenvironment characterization, researchers and clinicians can better understand vascular remodeling, drug distribution, and treatment response.

## **Why Use Erivessel?**
- **Quantitative Analysis**: Extracts precise vessel morphology and spatial distribution metrics.
- **Automated & Scalable**: Processes large datasets efficiently, reducing manual workload.
- **Interdisciplinary Integration**: Combines pathology and bioinformatic image analysis.
- **Enhances Drug Studies**: Assesses vascular changes induced by pharmacological treatments and their effect on drug penetration (e.g. in the current application Eribulin).
- **Open-Source & Modifiable**: Easily adaptable to different research contexts and imaging modalities.

## **How to Use Erivessel?**
1. **Prepare Your Data**:
   - 1.1 Obtain histological images of interest.
   - 1.2 Ensure the images are formatted correctly (e.g., `.tiff`, `.jpg`).
2. **QuPath Visualization of Slide & Tiling**
   - 2.1 Image visualization and annotation
   - 2.2 Tile creation
3. **Vessel Segmentation and Feature Extraction in Python**:
   - 3.1 Convertion of RGB to HSV format.
   - 3.2 Application of Otsu’s thresholding
   - 3.3 Stitching
   - 3.4 Morphological Operations
   - 3.5 Feature Extraction
4. **Interpret Your Data**

---

## LET'S START!
So, you've got your histological images, your burning scientific curiosity, and perhaps a steaming cup of coffee. Now what? Let's walk through the Erivessel pipeline together, breaking it down step by step. This isn't just a method; it's a journey—one that will take us from raw image data to profound insights into vascular remodeling. Buckle up!

### 1. **Prepare Your Data**
Before diving into analysis, we need to prepare your images. Here’s what you need to get your data ready for processing:

   - #### 1.1 Obtain histological images of interest.
     The first step is obtaining histological images of interest. These images must be stained for CD31, a marker for endothelial cells which line blood vessels. Slides need to be scanned digitally for analysis.

   - #### 1.2 Ensure the images are formatted correctly (e.g., `.tiff`, `.jpg`).
     Once you have your images, it’s crucial that they are in the right format for compatibility with the analysis pipeline. The most commonly used image formats for this type of work are .tiff (higher quality, slower processing) and .jpg (lower quality, faster processing).

### 2. **QuPath Visualization of Slide & Tiling**
To begin, you’ll need to load your whole-slide images into QuPath, a powerful image analysis software for digital pathology. In QuPath, you can zoom into specific regions, annotate areas of interest, and prepare your image for segmentation.
Tiling is a crucial preprocessing step for users with slide images that are too large to process efficiently, so they are divided into smaller, more manageable tiles. Vessels are captured with adequate detail while maintaining computational efficiency. 

 - #### 2.1 Image visualization and annotation
   - Open QuPath and load your histological slide.
   - Use QuPath’s built-in tools to annotate regions of interest (ROIs), marking vascular structures and tumor-adjacent areas.
   - Adjust visualization settings to enhance contrast and highlight vascular features for better segmentation.
   
 - #### 2.2 Tile creation
    - Navigate to: "Analyze → Tiles & Superpixels → Create Tiles" to generate sub-images.
    - Set tile size based on the expected vessel dimensions, ensuring adequate resolution for segmentation.
    - Execute the "ExportingImages.groovy" script to automate tile extraction, saving them as .jpg files.
    - Verify the tile output for completeness and quality before proceeding to segmentation.
   
```groovy
import qupath.lib.images.writers.ImageWriterTools
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

//How to make tiles
//First, Draw an annotation
//Then, Analyze --> Tiles & Superpixels --> Create Tiles
//You can define the size of your tiles
//Finally run the code.

def path = "Path/to/Slide"

def imageData = getCurrentImageData()
def server = getCurrentServer()
def filename = ""

def hierarchy = imageData.getHierarchy()

def annotations = hierarchy.getAnnotationObjects()
i = 1

//https://github.com/qupath/qupath/issues/62


for (annotation in annotations) {
    roi = annotation.getROI()
    
    print(annotation.getName())
    
    def request = RegionRequest.createInstance(imageData.getServerPath(),2, roi)
    
    tiletype = annotation.getParent().getPathClass()
    print(tiletype.toString())
    
    if (tiletype.toString().equals("Image")) {
    
        String tilename = String.format("200832.jpg")
    
        ImageWriterTools.writeImageRegion(server, request, path + "/" + tilename);
    
        //print("wrote " + tilename)
    
        
    }
    
    
   /* if (!tiletype.toString().equals("Image")) {
    
        String tilename = String.format("%s.jpg", annotation.getName())
    
        ImageWriterTools.writeImageRegion(server, request, path + "/" + tilename);
    
        //print("wrote " + tilename)
    
        i++
        
    }*/
}
```



### 3. **Vessel Segmentation and Feature Extraction in Python**
With the image tiles prepared, vessel segmentation and feature extraction can be performed using Python. 
- #### 3.1 Convertion of RGB to HSV format.
  The first step in this process is converting the extracted tiles from the standard RGB color space to HSV (Hue, Saturation, Value) format. This transformation is particularly useful because the H channel enhances the contrast of blood vessels, making them easier to distinguish from the surrounding tissue. Using OpenCV, the conversion is straightforward:

   ```python
   h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(h_s_v_image)
   h = 255 - h  # Inverting to enhance contrast
   ```

- #### 3.2 Application of Otsu’s thresholding
  Once the H channel is extracted, Otsu’s thresholding is applied to create a binary mask. This technique automatically determines an optimal threshold, separating vessel structures from the background:
  ```python
  ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
  binary = binary.astype(np.uint8) * 255
  ```
- #### 3.3 Stitching
- #### 3.4 Morphological Operations
  Morphological operations further refine the segmentation output. 
  Feel free to change the parameters below to fine tune the output to your necessities:
   
   ```python
   kernel_for_segmentation = (15, 15)
   kernel_for_closing = (15, 15)
   iterations_for_dilate = 4
   iterations_for_closing = 3
    ```

   Dilation is used to bridge small gaps between fragmented vessel structures, improving connectivity. Morphological closing, which involves dilation followed by erosion, eliminates small holes within vessels:
   ```python
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_for_segmentation)
   binary = cv2.dilate(binary, kernel, iterations=iterations_for_dilate)

   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_for_closing)
   opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations_for_closing)
   ```

   Contour detection is then employed to trace vessel boundaries, overlaying the detected vessels onto the original images for added interpretability:
   ```python
   contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   for c in contours:
    cv2.drawContours(original, [c], 0, (0, 255, 0), 5)
   cv2.imwrite(f"{folder}_Vessels.jpg", original)
   ```
- #### 3.5 Feature Extraction
   Once segmentation is complete, quantitative features of the detected vessels are extracted using `regionprops_table` from `skimage.measure`. These properties include:
   - `area`: The number of pixels inside each segmented vessel. It quantifies vessel size.
   - `axis_major_length`: The length of the longest axis of the vessel (major axis of the fitted ellipse).
   - `axis_minor_length`: The length of the shortest axis of the vessel (minor axis of the fitted ellipse).
   - `eccentricity`: Measures how elongated the vessel is (0 = perfect circle, 1 = line).
   - `orientation`: The angle (in degrees) of the major axis relative to the horizontal axis.

  ```python
   from skimage.measure import label, regionprops_table
   import pandas as pd

   label_img = label(opening)
   all_props = ["area", "axis_major_length", "axis_minor_length", "eccentricity", "orientation"]
   props = regionprops_table(label_img, original, properties=all_props)
   data = pd.DataFrame(props)
   data.to_csv(f'{folder}_Measurements.csv', index=True)
  ```
The output is a CSV file `{folder}_Measurements.csv` containing quantitative measurements of each detected vessel in the segmented image. The columns of the CSV file represent different morphological properties of the vessels, and each row corresponds to one labeled vessel.




![My Image](images/my_image.png)








### **1. Tile Extraction in QuPath (Groovy)**
This script extracts **tiles from whole-slide images** in **QuPath**, which are then used for vessel segmentation.

1. The script retrieves **annotations** (manually drawn regions of interest) from the image.
2. It creates **region requests** that extract specific areas of interest at a defined resolution.
3. If the annotation belongs to the "Image" category, it writes the extracted region as a **JPEG** file.
4. The extracted tiles are stored in the specified path for further processing.


### **2. Vessel Segmentation on a Single Image (Python)**
This script segments **vessels in a single image** using **OpenCV and skimage**.

1. **Loads an image** from the dataset.
2. **Converts it to HSV format**, extracting the **H-channel**, which provides better vessel contrast.
3. **Applies Otsu’s thresholding** to create a binary mask highlighting vessel regions.
4. **Uses morphological operations** to clean and enhance vessel structures.
5. **Labels detected vessels** and generates a colorized output.
6. **Saves the segmented image** for further analysis.


### **3. Batch Processing for Vessel Segmentation (Python)**
This script processes **multiple images in a batch**, automating segmentation across an entire dataset.

1. Reads **all image folders** in the dataset path.
2. Applies **the same vessel segmentation pipeline** as in Code 2.
3. Saves:
   - **Segmented images**.
   - **Extracted vessel features** (e.g., area, axis lengths, eccentricity, etc.) as a CSV file.
   - **Overlay images** with detected vessels marked.





## Ongoing
- **Gaussian process modelling**:
- **Fractal vessel statistical analyses**

## Conclusions
This pipeline integrates spatial vessel analysis into tumour microenvironment characterization, demonstrating the impact of vascular remodeling on drug penetration. The approach is generalizable and provides a foundation for investigating tumour vascularization, drug distribution, and treatment response.

## Repository
[GitHub Repository](https://github.com/slrenne/erivessel)

## Funding & Acknowledgments
This research was funded by Ricerca Finalizzata 2021 (Italian Ministry of Health - Giovani Ricercatori, Project Code GR-2021-12373209). 

Contributors: Dr. S. L. Renne, Dr. Ö. Mintemur, Dr. R. Frapolli, G. Grion, K. Roufail, F. E. Colella.

## Citation
If you use this pipeline, please cite:
> Renne et al., Integrating Spatial Vessel Analysis into Tumour Microenvironment Characterization, *Virchows Arch*, 485(Suppl 1), 2024.

## Contact
For inquiries, please reach out via [GitHub Issues](https://github.com/slrenne/erivessel/issues).
