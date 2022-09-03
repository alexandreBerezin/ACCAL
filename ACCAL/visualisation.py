## load probMatrix
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def saveResults(dataFolderPath,probLimit):

    matrixPath = pathlib.Path(dataFolderPath,"temp","probMat.npy")
    probMatrix = np.load(matrixPath)
    trilMatrix = np.tril(probMatrix,k=-1)

    sortedMat = np.argsort(-trilMatrix,axis=None)
    A,B = np.unravel_index(sortedMat,shape=np.shape(probMatrix))
    
    
    pathToSaveTo = pathlib.Path(dataFolderPath,"results")
    if pathToSaveTo.exists() == False:
            pathToSaveTo.mkdir()



    res = []
    for idx in range(len(A)):
        idA = A[idx]
        idB = B[idx]
        prob = trilMatrix[idA,idB]
        
        if prob>probLimit:
            res.append([idA,idB,prob])
            

    pathImgList = sorted(list(pathlib.Path(dataFolderPath,"temp","croppedImages").glob("*.png")))

    for idx,liaison in enumerate(res):

        id1 = liaison[0]
        id2 = liaison[1]
        prob = liaison[2]

        img1 = plt.imread(pathImgList[id1])
        img2 = plt.imread(pathImgList[id2])


        fig,axes = plt.subplots(1,2,figsize=(15,10))

        fig.suptitle(f"liaison n° {idx}  Probabilité = {prob:.2f} ",y=0.85)

        ax1,ax2 = axes
        ax1.axis("off")
        ax1.imshow(img1)
        ax1.set_title(pathImgList[id1].stem)


        ax2.imshow(img2)
        ax2.axis("off")
        ax2.set_title(pathImgList[id2].stem)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(pathlib.Path(pathToSaveTo,f"liason_{idx}.png"),bbox_inches='tight')

            