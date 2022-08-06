"""Image processing module
    
    Image processing module contains all the functions related to 
    image manipulation

    functions:
        - cropToCoin
        - processImage
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage

from pathlib import Path



def cropToCoin(imgPath:str)-> np.ndarray:
    """Return an cropped and centerd image.
    
    the size of the image is fixed to side of 360 pixels. The number of pixels 
    is important and will change the other parameters of the algorithm like
    the RBF kernel distance. The center is calculated with teh center of mass
    
    Args:
        imgPath (str): absolute path of the image

    Returns:
        np.ndarray: Array of pixels
    """
    #Read the image
    img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Cut 100 pixels under the image (legend)
    deltaL = 100
    img = img[:-deltaL,:]
    
    #Gray Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,thresh1 = cv2.threshold(gray,253,255,cv2.THRESH_BINARY)
    
    #Center of mass 
    mass_x, mass_y = np.where(thresh1 <= 0)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    
    ### Crop
    SidePixel = 360
    pixG = int(cent_x - SidePixel/2)
    pixD = int(cent_x + SidePixel/2)
    pixH = int(cent_y - SidePixel/2)
    pixB = int(cent_y + SidePixel/2)
    
    return img[pixG:pixD,pixH:pixB]


    
def processImage(img:np.ndarray,denoiseRatio:float,clipLimit:float)->np.ndarray:
    """multiple step image processing to get the contours
    
    the steps are : 
    - gray scale transformation
    - total variation denoise filter
    - CALHE contrast filter
    - another total variation denoise filter
    - sobel contour detection filter + normalisation to max = 1

    Args:
        img (np.ndarray): array of pixels RGB
        denoiseRatio (float): ratio for image denoise (= 0.1)
        clipLimit (float): clip limit for contrast tuning (=0.01)

    Returns:
        np.ndarray: array who represent the contour of the image
    """
    
    #gray transformation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    ## total variation 1
    imgTV1 = skimage.restoration.denoise_tv_chambolle(gray,denoiseRatio)
    
    ## CLAHE contrast
    imgCtr = skimage.exposure.equalize_adapthist(imgTV1, clip_limit= clipLimit)
    
    ## total variation 2 
    imgTV2 = skimage.restoration.denoise_tv_chambolle(imgCtr,denoiseRatio)
    
    ## contours filter
    
    #sobel contour
    out = skimage.filters.sobel(imgTV2)
    out = out/np.max(out) # normalise
    
    return out





def computeAndSaveTempImages(dataFolderPath:pathlib.Path,denoiseRatio:float,clipLimit:float)->None:
    """compute cropped and processed images and save them
    
    for each image (.png) file from a dataFolderPath, this function
    compute the cropped image and the contours of the cropped image 
    and then save them in the temp folder inside the dataFolder

    Args:
        dataFolderPath (pathlib.Path): Path of the data Folder
        denoiseRatio (float): ratio for image denoise (= 0.1)
        clipLimit (float): clip limit for contrast tuning (=0.01)
    """
    
    
    # get the list of all images in the folder
    pathList = list(dataFolderPath.glob("*.jpg"))
    
    ## Save temporary files in the data folder Path
    pathToSaveTo = Path(dataFolderPath,"temp")
    pathCroppedImages = Path(pathToSaveTo,"croppedImages")
    pathProcessedImages = Path(pathToSaveTo,"processedImages")

    ## if the Path does not exist create it :
    if pathToSaveTo.exists() == False:
        pathToSaveTo.mkdir()
        
    if pathCroppedImages.exists() == False:
        pathCroppedImages.mkdir()
        
    if pathProcessedImages.exists() == False:
        pathProcessedImages.mkdir()
        

    for path in pathList:

        name = path.stem

        imgC = cropToCoin(str(path))
        imgP = processImage(imgC,denoiseRatio=denoiseRatio,clipLimit=clipLimit)
        
        ## saves it in the lossless png
        savePathC = Path(pathCroppedImages,name+".png")
        plt.imsave(savePathC,imgC)

        savePathP = Path(pathProcessedImages,name+".png")
        plt.imsave(savePathP,imgP)


        