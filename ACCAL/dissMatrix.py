### Compute all pipeline for dissimilarity matrix 

import pathlib
from xml.dom.expatbuilder import theDOMImplementation
import scipy.sparse
import numpy as np

import modules.imageProcessing
import modules.features.kernel
import modules.features.selection
import modules.distance

## Parameters
dataFolderPath = pathlib.Path(r"D:\Stage\ACCAL\data\dataTest1")
appPath = pathlib.Path(r"D:\Stage\ACCAL\ACCAL")
DENOISE_RATIO = 0.1
CLIP_LIMIT = 0.01
LENGTH_KERNEL = 4.0
PIXEL_SIDE = 360

FEATURE_NUMBER = 250


def saveFeatures(dataFolderPath,absAppPath,pixelSide):
    print("Saving Features : ")
    processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
    pathList = sorted(list(processedImgPath.glob("*npy")))
    n = len(pathList)
    
    K = modules.features.kernel.getK(l=LENGTH_KERNEL,absAppPath=absAppPath,pixelSide=pixelSide)
    
    
    for idx,path in enumerate(pathList):
        print(f"{idx}/{n}",end='\r')
        _ = modules.features.selection.getFeatures(nbFeatures=FEATURE_NUMBER,K=K,imgPath=path)
    
    print("")
    print("Done")
    print("")


def saveDissMatrix(dataFolderPath):
    
    print("Computing pairwise Distances : ")
    
    processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
    pathList = sorted(list(processedImgPath.glob("*.npy")))
    nameList = [ path.stem for path in pathList]
    nb = len(nameList)

    Nb = np.zeros((nb,nb))
    P = np.zeros((nb,nb))

    ### Computing paiwise distance

    tempPath = pathlib.Path(dataFolderPath,"temp")
    Nb = np.zeros((nb,nb))
    P = np.zeros((nb,nb))
    
    for idx,name1 in enumerate(nameList):
        print(f"{idx}/{nb}",end='\r')
        img1,coord1 = modules.distance.getImgFeat(tempPath,name1)
        for idx2 in range(idx+1,nb):
            img2,coord2 = modules.distance.getImgFeat(tempPath,nameList[idx2])
            kp1,kp2,match = modules.distance.getFilteredMatch(img1,coord1,img2,coord2,3)
            
            [nbMatch,p] = modules.distance.getNbAndP(kp1,kp2,match)
            Nb[idx2,idx] = nbMatch
            P[idx2,idx] = p
            
    print("")
    
            

    # Processing of the matrix
    Nb = Nb.transpose() + Nb
    P = P.transpose() + P
    

    np.save(pathlib.Path(tempPath,"Nb"),Nb)
    np.save(pathlib.Path(tempPath,"P"),P)

    np.fill_diagonal(Nb,1)
    np.fill_diagonal(P,1)

    A = 1/Nb
    B = np.log(P) 
    
    np.fill_diagonal(A,np.nan)
    np.fill_diagonal(B,np.nan)
    
    ## normalisation step
    minA = np.nanmin(A)
    A = A -minA
    maxA = np.nanmax(A)
    A = A/(2*maxA)

    minB = np.nanmin(B)
    B = B-minB
    maxB = np.nanmax(B)
    B = B/(2*maxB) 


    np.fill_diagonal(A,0)
    np.fill_diagonal(B,0)

    ## Distance Matrix
    D = A+B

    np.save(pathlib.Path(tempPath,"distMatrix"),D)


#modules.imageProcessing.computeAndSaveTempImages(dataFolderPath=dataFolderPath,denoiseRatio=DENOISE_RATIO,clipLimit=CLIP_LIMIT)


#saveFeatures(dataFolderPath=dataFolderPath,absAppPath=appPath,pixelSide=PIXEL_SIDE)


saveDissMatrix(dataFolderPath=dataFolderPath)
