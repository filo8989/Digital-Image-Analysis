# Integrating Spatial Vessel Analysis into Tumour Microenvironment Characterization

## Overview
Understanding the tumour microenvironment is crucial for targeted therapies. While various cellular components are extensively studied, spatial vessel analysis is often overlooked. This project presents a pipeline for vessel segmentation and analysis, applied to evaluate whether eribulin (ERI) treatment improves docetaxel (DTX) distribution in leiomyosarcoma (LMS) through vascular remodeling.

## Research Questions
- Does Eribulin treatment alter vessel characteristics?
- How does vessel distribution impact drug penetration and distribution?

## Methods
1. **Sample Preparation**: LMS patient-derived xenograft (PDX) model slices stained for CD31 (endothelial marker) to visualize vessels.
2. **Image Processing**:
   - TIFF whole-slide images acquired.
   - Tiling performed using Groovy and QuPath for quality control.
   - RGB-to-HSV conversion in Python, using the H-channel for segmentation.
   - Morphological operations for vessel detection.
3. **Data Extraction**:
   - Vessel mask generation.
   - Measurements: area, major/minor axis, eccentricity, orientation.
4. **Spatial Analysis & Imaging Mass Spectrometry (IMS)**:
   - Vessel density maps created.
   - IMS data superimposed to assess intra-tumoral drug diffusion.

## Implementation

### **1. Tile Extraction in QuPath (Groovy)**
This script extracts **tiles from whole-slide images** in **QuPath**, which are then used for vessel segmentation.

#### **How It Works:**
1. The script retrieves **annotations** (manually drawn regions of interest) from the image.
2. It creates **region requests** that extract specific areas of interest at a defined resolution.
3. If the annotation belongs to the "Image" category, it writes the extracted region as a **JPEG** file.
4. The extracted tiles are stored in the specified path for further processing.

```groovy
import qupath.lib.images.writers.ImageWriterTools
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

def path = "D:/Ribulin_Tiff_Files/200832"

def imageData = getCurrentImageData()
def server = getCurrentServer()
def hierarchy = imageData.getHierarchy()

def annotations = hierarchy.getAnnotationObjects()

for (annotation in annotations) {
    roi = annotation.getROI()
    def request = RegionRequest.createInstance(imageData.getServerPath(), 2, roi)
    tiletype = annotation.getParent().getPathClass()
    if (tiletype.toString().equals("Image")) {
        String tilename = "200832.jpg"
        ImageWriterTools.writeImageRegion(server, request, path + "/" + tilename);
    }
}
```

### **2. Vessel Segmentation on a Single Image (Python)**
This script segments **vessels in a single image** using **OpenCV and skimage**.

#### **How It Works:**
1. **Loads an image** from the dataset.
2. **Converts it to HSV format**, extracting the **H-channel**, which provides better vessel contrast.
3. **Applies Otsu’s thresholding** to create a binary mask highlighting vessel regions.
4. **Uses morphological operations** to clean and enhance vessel structures.
5. **Labels detected vessels** and generates a colorized output.
6. **Saves the segmented image** for further analysis.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import measure, color
import os

original = cv2.imread("D:/Humanitas_Files/Vessel_Segmentation_and_Python_Files/Ribulin_Tiff_Files/195320/195320.jpg")
h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(h_s_v_image)
h = 255 - h
ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
binary = cv2.dilate(binary, kernel, iterations=4)

label_img = label(binary)
img = color.label2rgb(label_img, bg_label=0)
plt.imsave("D:/Humanitas_Files/Segmented.jpg", img)
```

### **3. Batch Processing for Vessel Segmentation (Python)**
This script processes **multiple images in a batch**, automating segmentation across an entire dataset.

#### **How It Works:**
1. Reads **all image folders** in the dataset path.
2. Applies **the same vessel segmentation pipeline** as in Code 2.
3. Saves:
   - **Segmented images**.
   - **Extracted vessel features** (e.g., area, axis lengths, eccentricity, etc.) as a CSV file.
   - **Overlay images** with detected vessels marked.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import measure
import pandas as pd
import os

PATH = "D:/Ribulin_Tiff_Files"
folders = [item for item in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, item))]

for folder in folders:
    general_path = os.path.join(PATH, folder)
    image_path = general_path + "/" + folder + ".jpg"
    original = cv2.imread(image_path)

    h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(h_s_v_image)
    h = 255 - h
    ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)

    label_img = label(binary)
    props = measure.regionprops_table(label_img, original, properties=["area", "axis_major_length", "axis_minor_length", "eccentricity", "orientation"])
    data = pd.DataFrame(props)
    data.to_csv(f'{general_path}/{folder}_SEGMENTATION/Measurements_{folder}.csv', index=True)
```

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
