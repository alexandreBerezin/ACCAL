"""Image processing module
    
    Image processing module contains all the functions related to 
    image manipulation

    functions:
        - cropToCoin
        - processImage
"""

import numpy as np
import cv2
import skimage



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



################################################################
######### A effacer ################

from skimage.restoration import denoise_tv_chambolle

def preprocess(img :np.ndarray,flou:int,resize:int=0)->np.ndarray:
    '''Renvoie une image apres
    - flou gaussien
    - detecteur de contours (Laplace)'''

    img = cv2.GaussianBlur(img,(flou,flou),cv2.BORDER_DEFAULT)
    contours = cv2.convertScaleAbs(cv2.Laplacian(img,cv2.CV_64F))
    if(resize != 0):
        contours = cv2.resize(contours, (resize,resize), interpolation = cv2.INTER_AREA)
    contours = np.float32(contours)
    return contours
    


    
    
def getContour(img:np.ndarray,denoiseTVWeight:float)->np.ndarray:
    #transformation en nuances de gris 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Total variation regularization
    denoisedImg = denoise_tv_chambolle(gray,weight=denoiseTVWeight)

    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    denoisedImg = cv2.normalize(src=denoisedImg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cl1 = clahe.apply(denoisedImg)
    
    #Laplace
    Laplace = cv2.Laplacian(cl1,cv2.CV_64F)
    contours = cv2.convertScaleAbs(Laplace)

    #Normalisation
    norm = np.linalg.norm(contours)
    contours = contours/norm
    
    return contours

