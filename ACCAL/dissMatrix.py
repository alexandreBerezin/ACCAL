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
dataFolderPath = pathlib.Path(r"D:\Stage\ACCAL\data\dataTest2")
appPath = pathlib.Path(r"D:\Stage\ACCAL\ACCAL")
DENOISE_RATIO = 0.1
CLIP_LIMIT = 0.01
LENGTH_KERNEL = 4.0
PIXEL_SIDE = 360

FEATURE_NUMBER = 250




#modules.imageProcessing.computeAndSaveTempImages(dataFolderPath=dataFolderPath,denoiseRatio=DENOISE_RATIO,clipLimit=CLIP_LIMIT)

K = modules.features.kernel.getK(l=LENGTH_KERNEL,absAppPath=appPath,pixelSide=PIXEL_SIDE)

processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
pathList = sorted(list(processedImgPath.glob("*npy")))
for idx,path in enumerate(pathList):
    print(idx)
    _ = modules.features.selection.getFeatures(nbFeatures=FEATURE_NUMBER,K=K,imgPath=path)

print("DONE")

# path
pathList = sorted(list(processedImgPath.glob("*.npy")))
nameList = [ path.stem for path in pathList]
nb = len(nameList)


Nb = np.zeros((nb,nb))
P = np.zeros((nb,nb))



################################

print("Compute pairwise distance ")

tempPath = pathlib.Path(dataFolderPath,"temp")


Nb = np.zeros((nb,nb))
P = np.zeros((nb,nb))

for idx,name1 in enumerate(nameList):
    print(idx)
    img1,coord1 = modules.distance.getImgFeat(tempPath,name1)
    for idx2 in range(idx+1,nb):
        img2,coord2 = modules.distance.getImgFeat(tempPath,nameList[idx2])
        
        kp1,kp2,match = modules.distance.getFilteredMatch(img1,coord1,img2,coord2,3)
        
        [nbMatch,p] = modules.distance.getNbAndP(kp1,kp2,match)
        
        Nb[idx,idx2] = nbMatch
        P[idx,idx2] = p
        


Nb = Nb.transpose() + Nb
P = P.transpose() + P

np.fill_diagonal(Nb,1)
np.fill_diagonal(P,1)

A = 1/Nb
B = np.log(P) 


## normalisation step
minA = np.min(A)
maxA = np.max(A[A!=np.inf])

minB = np.min(B[B!=-np.inf])
maxB = np.max(B)


A = A - minA
A = A/2*np.max(A[A!=np.inf])

B = B-minB
B = B/2*np.max(B)

np.fill_diagonal(A,0)
np.fill_diagonal(B,0)

## Distance Matrix
D = A+B

np.save(pathlib.Path(tempPath,"distMatrix"),D)

