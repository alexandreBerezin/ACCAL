import pathlib
import scipy.sparse
import numpy as np


def getK(l:float,absAppPath:pathlib.Path,pixelSide:int)-> scipy.sparse.lil_matrix:
    """Compute and save or retrieve K (kernel) Matrix 

    Args:
        l (float): caracteristic lenght of the RBF kernel
        absAppPath (pathlib.Path): absolute path of the app
        pixelSide (int): number of pixels on the side of the picture

    Returns:
        scipy.sparse.lil_matrix: K matrix (sparse matrix) 
    """

    ## Search for a precomputed value of K(l,pixelSide)    
    absKDataPath = pathlib.Path(absAppPath,"modules","features","kernelData")
    
    pathList = list(absKDataPath.glob("l(%.2f)PxSide(%d)*"%(l, pixelSide)))
    if len(pathList) > 0 :
        return scipy.sparse.load_npz(pathList[0]).astype(np.float32,copy = False)
        
    
    ## If the K matrix is not already saved 
    ## compute new K matrix
    K = computeK(l=l,pixelSide=pixelSide)
    savePath = pathlib.Path(absKDataPath, "l(%.2f)PxSide(%d)"%(l, pixelSide) )
    
    scipy.sparse.save_npz(str(savePath),K)
    return K



def computeK(l:float,pixelSide:int)->scipy.sparse.lil_matrix:
    """Compute an approximate version of the K matrix 
    
    Based on finding the closest neghboors of each pixels

    Args:
        l (float): caracteristic lenght of the RBF kernel
        pixelSide (int): number of pixels on the side of the picture

    Returns:
        scipy.sparse.lil_matrix: K (kernel) matrix 
    """
    
    vecSize = pixelSide*pixelSide
    M = scipy.sparse.lil_matrix((vecSize,vecSize),dtype=np.float32)
    
    nbNeigb = int(np.floor(3*l))
    
    for i in range(vecSize):
        setN = getNeighbours(i,nbNeig=nbNeigb ,pixelSide=pixelSide)
        for j in setN:
            M[i,j] = np.exp(-dist2(i,j,pixelSide=pixelSide)/(2*l**2))
            
    return M.tocsr()




############################################
    
def dist2(i:int,j:int,pixelSide:int):
    # some arithmetics to take de suare distance in a grid
    return (j%pixelSide-i%pixelSide)**2 + (j//pixelSide - i//pixelSide)**2 
    

def getNeighbours(i:int,nbNeig:int,pixelSide:int)->set:
    
    
    y,x=divmod(i,pixelSide)
    
    setN = set()
    for k in range(-nbNeig,+nbNeig+1):
        for l in range(-nbNeig,+nbNeig+1):
            xc = x+ k
            yc = y+ l
            if (xc>=0 and xc<pixelSide) and (yc>=0 and yc<pixelSide):
                setN.add(pixelSide*yc+xc)
    return setN