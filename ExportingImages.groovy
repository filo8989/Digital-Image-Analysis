import qupath.lib.images.writers.ImageWriterTools
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

//How to make tiles
//First, Draw an annotation
//Then, Analyze --> Tiles & Superpixels --> Create Tiles
//You can define the size of your tiles
//Finally run the code.

def path = "D:/Ribulin_Tiff_Files/200832"

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