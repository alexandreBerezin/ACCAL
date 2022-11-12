#### Functions gloabl
import numpy as np

def getRhoFromZ(Z:np.ndarray)->list:
    '''
    return the list rho from array Z
    rho = [{},{}, .. ]
    a list of sets
    '''
    nbClust = np.max(Z)+1
    rho = []
    for i in range(nbClust):
        rho.append(set(np.argwhere(Z==i).ravel()))
    return rho

def getZfromRho(rho):
    return 0


def getHfromD(D:np.ndarray)->np.ndarray:
    '''
    Return h, an array used for histogram,
    containing all the values different from 0
    from the array D
    '''
    trilMat = np.copy(D)
    trilMat[np.triu_indices(np.shape(trilMat)[0])] = np.nan
    histVec = trilMat.ravel()
    return histVec[~np.isnan(histVec)]


def getLimitFromH(h:np.ndarray,proportion:float)->float:
    '''
    return the limit to seperate in groups A and B 
    according to the proportion a priori of die link
    '''
    h= np.sort(h)
    l, = np.shape(h)
    nb = int(proportion*l)
    return h[nb]