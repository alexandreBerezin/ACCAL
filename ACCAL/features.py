import pathlib
import modules.features.kernel
import modules.features.selection
import sys

import multiprocessing as mp 

print("START FEATURE APP : ", __name__)

_,dataFolderPath,absAppPath,pixelSide,lengthKernel,featureNumber,cores = sys.argv


dataFolderPath = pathlib.Path(dataFolderPath)
absAppPath = pathlib.Path(absAppPath)
pixelSide = int(pixelSide)
lengthKernel = float(lengthKernel)
featureNumber = int(featureNumber)
cores = int(cores)

    
K = modules.features.kernel.getK(l=lengthKernel,absAppPath=absAppPath,pixelSide=pixelSide)

if __name__ == "__main__":
    
    processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
    pathList = sorted(list(processedImgPath.glob("*npy")))
    n = len(pathList)
    processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
        
    pool = mp.Pool(cores)
    
    for idx,path in enumerate(pathList):
        pool.apply_async(modules.features.selection.getFeatures, (featureNumber,K,path))
    pool.close()
    pool.join()




