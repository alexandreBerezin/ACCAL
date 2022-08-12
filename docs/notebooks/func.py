import numpy as np
import scipy.sparse

from pathlib import Path

from multiprocessing import shared_memory



def getKwColumn(K:scipy.sparse.csr_matrix,imgVec:np.ndarray,column:int)->scipy.sparse.csr_matrix:
    KD = K[:,column].multiply(imgVec.reshape(-1,1))
    return K.transpose().dot(KD)



def getFeatures(nbFeatures:int,sharedMem:list,Kshape:tuple, imgPath:Path)->list:
    
    
    shm1 = shared_memory.SharedMemory(name=sharedMem[0][2])
    shm2 = shared_memory.SharedMemory(name=sharedMem[1][2])
    shm3 = shared_memory.SharedMemory(name=sharedMem[2][2])
    
    data = np.ndarray(sharedMem[0][0],dtype=sharedMem[0][1],buffer = shm1.buf)
    indices = np.ndarray(sharedMem[1][0],dtype=sharedMem[1][1],buffer = shm2.buf)
    indptr = np.ndarray(sharedMem[2][0],dtype=sharedMem[2][1],buffer = shm3.buf)
    
    K = scipy.sparse.csc_matrix( (data,indices,indptr), Kshape)
    
    print(f"début {imgPath.name}") 
    
    img = np.load(imgPath)
    imgVec = img.ravel().reshape(-1,1)
    
    K2 = K.power(2)
    T = K2.transpose()
    traceVec = T.dot(imgVec.reshape(-1,1))
    
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
            invVarXsiDiag = scipy.sparse.diags(invVarXsi)


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
    return featuresList

        
    
def getFeatures(nbFeatures:int,K,imgPath:str):
    
    img = np.load(imgPath)
    imgVec = img.ravel().reshape(-1,1)
    
    K2 = K.power(2)
    T = K2.transpose()
    traceVec = T.dot(imgVec.reshape(-1,1))
    
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
            invVarXsiDiag = scipy.sparse.diags(invVarXsi)


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
    return featuresList
