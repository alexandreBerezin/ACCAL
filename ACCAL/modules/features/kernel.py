from importlib.resources import path
import os
import pathlib
import scipy

import numpy as np


def getK(nbSide:int,l:float,absAppPath:pathlib.Path)-> scipy.sparse.lil_matrix   :
    """Get K matrix for the with l kernel length


    Args:
        nbSide (int): number of pixels on the side 
        l (float): caracteristic length of the RBF kernel
        absPath (str): _description_

    Returns:
        scipy.sparse.lil_matrix: K sparse matrix
    """
    
    
    ## Search for a precomputed value of K(l)
    
    absKDataPath = pathlib.Path(absAppPath,"features","kernelData")
    
    pathList = list(absKDataPath.glob("L_*"))
    for path in pathList:
        if str(l) in path.name:
            return scipy.sparse.load_npz(path).astype(np.float32)
            
    
    ## If the K matrix is not already saved 
    ## compute new K matrix
    #print("Calcul de K")
    #K = computeK(nbSide,l)
    #print("sauvegarde de K")
    #s.save_npz(absKFolderPath + '/L_%f'%l, K)
    return 0 



#########################################################

def getVoisins(idx:int,nbSide:int,d:int):
    '''
    renvoie la liste des indices voisins de idx
    ----------------------
    idx: index du pixel dans un vecteur 1D
    nbSide: nombre de pixel de coté 
    d : distance du plus grand voisins '''
    
    #Coordonnées du centre
    Cx,Cy = getCoordFromVect(idx,nbSide)
    #Coordonnées de l'origine
    Ox = Cx-d
    Oy = Cy-d
    voisins = []
    for x in range(Ox,Ox+2*d+1):
        for y in range(Oy,Oy+2*d+1):

            # Valeurs dans la grille ? 
            if ((x>=0 and x<nbSide) and (y>=0 and y <nbSide)):
                voisins.append([x,y])
                
    
    ## Tranformation en idices 1D
    nb = len(voisins)
    idxVoisins = np.zeros(nb)
    for coord in range(nb):
        i,j = voisins[coord]
        idxVoisins[coord] = getIdxFromArray(i,j,nbSide)
    
    return idxVoisins




def RBF(a:np.ndarray,b:np.ndarray,l:float):
    xa,ya = a
    xb,yb = b
    d2 = (xa-xb)**2 + (ya-yb)**2
    return np.exp(-d2/(2*l**2),dtype=np.float32)




def computeK(nbSide:int,l:float)-> s.lil.lil_matrix:
    '''
    Calcule et renvoie la matrice K approchée RBF
    avec un cube de longeur  pour chaque 
    ------------------------------
    nbSide: nombre de pixel de coté 
    l: longeur caractéristique du RBF
    '''
    d = int(np.floor(3*l))
    N = nbSide*nbSide
    M = s.lil_matrix((N,N),dtype=np.float32)
    

    for i in range(N):
        voisins = getVoisins(i,nbSide,d)
        x = getCoordFromVect(i,nbSide)
        for j in voisins:
            y = getCoordFromVect(j,nbSide)
            M[i,j] = RBF(x,y,l)
            
    return s.csc_matrix(M).astype(np.float32,copy=False)



###################################

def getCoordFromVect(idx:int,nbSide:int)-> np.ndarray:
    '''
    renvoie les coordonnées du pixel en fonction de 
    son indice dans un tableau 1D 
    --------------------------
    idx : index dand le tableau 1D
    nbSide : nombre de pixel sur le coté de l image carrée
    '''
    x = idx//(nbSide)
    y = idx%(nbSide)
    return np.array([x,y]) 
    
def getIdxFromArray(i:int,j:int,nbSide:int)->int:
    '''
    renvoie l'index du pixel dont les coordonnées sont 
    i et j 
    --------------------------
    i,j : coordonnées
    nbSide : nombre de pixel sur le coté de l image carrée
    '''
    return i*nbSide+ j 


def getCoordFromVectList(liste :list,nbSide:int)->np.ndarray:
    out = []
    for idx in liste:
        x,y = getCoordFromVect(idx,nbSide)
        out.append([x,y])
    
    return np.array(out)