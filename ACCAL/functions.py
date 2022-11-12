### Definition of the main fnuctions used by the algorithm 

import pathlib
import scipy.sparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt


import modules.features.kernel
import modules.features.selection
import modules.distance

import modules.sampling
import modules.clusters

import modules.general
import modules.groups
import modules.mcmc

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
    
def clustering(dataFolderPath,ratioLow,ratioHigh,nbBurnin,nbSample,eachSample):
    
        
    s = f""" 
CLUSTERING STEP
{"-"*40}
data path : {dataFolderPath}
"""   
    print(s)

    # load the dissMatrix
    matrixPath = pathlib.Path(dataFolderPath,"temp","distMatrix.npy")
    D = np.load(matrixPath) 
    
    h = modules.general.getHfromD(D)
    

    limB = modules.general.getLimitFromH(h,ratioLow)
    limH = modules.general.getLimitFromH(h,ratioHigh)
    
    limit = (limB+limH)/2
    
    

    hierarircalModel = AgglomerativeClustering(n_clusters=None,affinity='precomputed',linkage='single',distance_threshold=limit)
    res = hierarircalModel.fit(D)
    Z = res.labels_
    
    rho = modules.general.getRhoFromZ(Z)
    
    A = h[h<= limH]
    B = h[h>  limB]
    
    print("limit H ",limH)
    print("limit B ",limB)
    
    
    
    bins = np.linspace(0,np.max(h),50)
    plt.hist(B,bins,histtype='step', stacked=True, fill=False,log=True)
    plt.hist(A,bins,histtype='step', stacked=True, fill=False,log=True)
    plt.savefig(pathlib.Path(dataFolderPath,"temp","disrib.png"))
    

    


    delta1,alpha,beta = modules.groups.fitValues(A,alpha0=1,beta0=10,burnin=1000,nbSample=50,deltaSample=50)
    delta2, zeta ,gamma  = modules.groups.fitValues(B,alpha0=15,beta0=20,burnin=1000,nbSample=50,deltaSample=50)


    ## valeur initiale
    p = 0.5
    r = 2

    N,_ = np.shape(D)
    r=2
    p=0.5
    
    print("")

    print("final burnin...")

    for _ in range(nbBurnin):
        print(f"{_}/{nbBurnin}",end='\r')
        r = modules.mcmc.drawSampleMCMCforR(priorR=r,p=p,rho=rho,etha = 1,sigma = 1)
        
        p = modules.mcmc.drawSampleforP(r= r,rho=rho,N=N,u=1,v=1)
        
        for i in range(N):
            rho,Z = modules.mcmc.drawSampleRhoI(rho,D,Z,i,p,r,delta1,alpha,beta,delta2, zeta ,gamma)


    print("")
    print("final sampling ...")
    # Calcul final 
    samples = []
    for i in range(nbSample):
        print(f"{i}/{nbSample}",end='\r')
        r = modules.mcmc.drawSampleMCMCforR(priorR=r,p=p,rho=rho,etha = 1,sigma = 1)
        
        p = modules.mcmc.drawSampleforP(r= r,rho=rho,N=N,u=1,v=1)
        
        for j in range(N):
            rho,Z = modules.mcmc.drawSampleRhoI(rho,D,Z,j,p,r,delta1,alpha,beta,delta2, zeta ,gamma)

            
        if i%eachSample == 0:
            samples.append(np.copy(Z))

    print("Sampling Done, Saving results ...")
    probMat = modules.sampling.getProbMatrix(samples)
    np.save(pathlib.Path(dataFolderPath,"temp","probMat.npy"),probMat)



