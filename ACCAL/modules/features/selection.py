import scipy.sparse
import numpy as np
import pathlib


def getFeatures(nbFeatures:int,K,imgPath:pathlib.Path)-> np.ndarray:
    """returns nbFeatures from the path of the processed image

    Args:
        nbFeatures (int): number of Features
        K (_type_): K(l) (kernel) matrix to use 
        imgPath (pathlib.Path): _Path to the image in the processedImage folder

    Returns:
        np.ndarray: an array of indexes of teh features in vec format 
    """

    featureFolder = pathlib.Path(imgPath.parent.parent,"features")
    res = list(featureFolder.glob(imgPath.stem+"*"))
    if len(res) != 0:
        return np.load(res[0])
        
    img = np.load(imgPath)
    imgVec = img.ravel().reshape(-1,1)
    
    traceVec = K.power(2).transpose().dot(imgVec.reshape(-1,1))
    
    ## Initial values
    featuresList = []
    varVec = 0
    psi = 0


    for i in range(nbFeatures):
        # first feature
        if i == 0 :
            ## First feature
            idxMax = np.argmax(imgVec)
            featuresList.append(idxMax)

            col = getKwColumn(K,imgVec,idxMax)
            psi = scipy.sparse.csr_matrix(col)

            varXsi = traceVec[featuresList].ravel()
            invVarXsi = 1/varXsi

            varVec = scipy.sparse.csc_matrix(traceVec) - psi.power(2).dot(invVarXsi.reshape(-1,1)).reshape(-1,1)

        else: 
            idxMax = np.argmax(varVec)
            featuresList.append(idxMax)
            
            col = getKwColumn(K,imgVec,idxMax)
            psi = scipy.sparse.hstack([psi,col])

            varXsi = traceVec[featuresList].ravel()
            invVarXsi = 1/varXsi
            varVec = scipy.sparse.csc_matrix(traceVec) - psi.power(2).dot(invVarXsi.reshape(-1,1)).reshape(-1,1)
            
    # end of the loop 
    np.save(pathlib.Path(featureFolder,imgPath.name),featuresList)
    return np.array(featuresList)




def getKwColumn(K:scipy.sparse.csr_matrix,imgVec:np.ndarray,column:int)->scipy.sparse.csr_matrix:
    """Compute a column of Kw = K^T * W * K matrix

    Args:
        K (scipy.sparse.csr_matrix): K matrix
        imgVec (np.ndarray): vector of a processed image
        column (int): column

    Returns:
        scipy.sparse.csr_matrix: column vector of Kw
    """
    KD = K[:,column].multiply(imgVec.reshape(-1,1))
    return K.transpose().dot(KD)
