### Definition of the main fnuctions used by the algorithm 

import pathlib
import scipy.sparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering



import modules.features.kernel
import modules.features.selection
import modules.distance

import modules.sampling
import modules.clusters

## Parameters



#### clustering

def saveFeatures(dataFolderPath,absAppPath,pixelSide,lengthKernel,featureNumber):
    print("Saving Features : ")
    processedImgPath = pathlib.Path(dataFolderPath,"temp","processedImages")
    pathList = sorted(list(processedImgPath.glob("*npy")))
    n = len(pathList)
    
    K = modules.features.kernel.getK(l=lengthKernel,absAppPath=absAppPath,pixelSide=pixelSide)
    
    for idx,path in enumerate(pathList):
        print(f"{idx}/{n}",end='\r')
        _ = modules.features.selection.getFeatures(nbFeatures=featureNumber,K=K,imgPath=path)
    
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
    return 0
    
def clustering(dataFolderPath,distCut,nbBurnin,nbSample,eachSample):

    # load the dissMatrix
    matrixPath = pathlib.Path(dataFolderPath,"temp","distMatrix.npy")
    dissMatrix = np.load(matrixPath) 

    # creating an initial vector for the sampling
    trilMat = np.copy(dissMatrix)
    trilMat[np.triu_indices(np.shape(trilMat)[0])] = np.nan
    
    histVec = trilMat.ravel()
    histVec = np.array(sorted(histVec[~np.isnan(histVec)]))


    #### Take the 1% Lower liks
    #nbCut = int(len(histVec)*0.01)
    #distCut = histVec[nbCut]

    ## OR empirical value : ##
    distCut = 0.2


    hierarircalModel = AgglomerativeClustering(n_clusters=None,affinity='precomputed',linkage='single',distance_threshold=distCut)
    res = hierarircalModel.fit(dissMatrix)
    Zinit = res.labels_


    #### Gibbs sampling

    ### Hypermarameters

    # for r
    modules.sampling.etha_ =2
    modules.sampling.sigma_ =1

    # for p 
    modules.sampling.u_ = 1
    modules.sampling.v_ = 1

    # for Z
    A = histVec[histVec<= distCut]
    B = histVec[histVec>distCut]
    
    # si aucune liaison dans A --> pas de liason de coin
    lenA, = np.shape(A)
    if lenA == 0 :
        print("ERREUR : peut être trop peu de monnaies ")
        return 0 

    ### fit value
    deltaA,alphaA,betaA = modules.clusters.fitValues(A)
    deltaB,alphaB,betaB = modules.clusters.fitValues(B)


    modules.sampling.delta1,modules.sampling.alpha_ ,modules.sampling.beta_  = deltaA,alphaA,betaA
    modules.sampling.delta2, modules.sampling.xsi_ ,modules.sampling.gamma_  = deltaB,alphaB,betaB



    Z = np.copy(Zinit)

    ## valeur initiale
    p=0.5
    r = 3


    n,_ = np.shape(dissMatrix)
    D = np.copy(dissMatrix)




    ### burnin 
    nbBurnin = 20

    for i in range(nbBurnin):
        print(f"{i}/{nbBurnin}",end='\r')
        #Draw r
        r = modules.sampling.drawSampleMCMCforR(r,p,Z)
        #Draw p
        p = modules.sampling.drawSampleforP(r,Z,n)
        #Draw Z
        for i in range(n):
            Z = modules.sampling.sampleNewZForI(i,p,r,Z,D)
            
            
            

    # Calcul final 
    samples = []
    for i in range(nbSample):
        print(f"{i}/{nbSample}",end='\r')
        #Draw r
        r = modules.sampling.drawSampleMCMCforR(r,p,Z)
        #Draw p
        p = modules.sampling.drawSampleforP(r,Z,n)
        #Draw Z
        for j in range(n):
            Z = modules.sampling.sampleNewZForI(j,p,r,Z,D)
            
        if i%eachSample == 0:
            samples.append(Z)


    probMat = modules.sampling.getProbMatrix(samples)
    np.save(pathlib.Path(dataFolderPath,"temp","probMat.npy"),probMat)



