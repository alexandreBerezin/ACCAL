import numpy as np
import cv2
import re


def cropToCoin(imgPath:str)-> np.ndarray:
    #Lecture
    img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Couper le bas 100 pixel
    deltaL = 100
    img = img[:-deltaL,:]
    
    #Convesrion en gris
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,thresh1 = cv2.threshold(gray,253,255,cv2.THRESH_BINARY)
    
    #Calcul du centre de masse
    mass_x, mass_y = np.where(thresh1 <= 0)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    
    ### Recadrage
    SidePixel = 360
    pixG = int(cent_x - SidePixel/2)
    pixD = int(cent_x + SidePixel/2)
    pixH = int(cent_y - SidePixel/2)
    pixB = int(cent_y + SidePixel/2)
    
    return img[pixG:pixD,pixH:pixB]
    

    
    
def getPathFromLiaisons(liaisons:list,dataPath:list)->list:
    pathLiaison = []
    
    idList = []

    for l in liaisons:
        idCoin = re.findall(r'\d+', l)
        subArray = []
        subArrayId = []
        for i in idCoin:
            for idx,path in enumerate(dataPath):
                if i in path:
                    subArray.append(path)
                    subArrayId.append(idx)
        pathLiaison.append(subArray)
        idList.append(subArrayId)
        
    return pathLiaison,idList
                
    