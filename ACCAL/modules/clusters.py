from typing import Tuple
import numpy as np 



def plotClusters(ax,data,Z):
    K = np.max(Z)+1
    for i in range(K):
        datai = data[np.where(Z==i)]
        ax.scatter(datai[:,0],datai[:,1],label="cluster %d"%i)
    return ax


def getSimulatedData()->Tuple[np.ndarray,np.ndarray]:
    mu = np.zeros((15,2))

    for i in range(15):
        mu[i,:]=np.random.rand(2)*100

    data = []

    sig = 5

    cov = np.array([[sig,0],
                   [0,sig]])

    for i in range(15):
        ni = np.random.negative_binomial(2,0.8)+1
        for _ in range(ni):
            data.append(list(np.random.multivariate_normal(mu[i,:],cov)))

    data = np.array(data)
    
    def getDistMatrix(data):
        n,_ = np.shape(data)
        
        D = np.zeros((n,n))
        
        for i in range(n):
            for j in range(n):
                xi,yi = data[i]
                xj,yj = data[j]
                
                D[i,j] = np.sqrt((xi-xj)**2 + (yi-yj)**2)
        return D
    
    D = getDistMatrix(data)
    return data,D 


####VIsualisation distrib 

from scipy.special import gamma,gammaln

def logPdfHyper(x,delta1,alpha,beta):
    logC = alpha*np.log(beta) + gammaln(delta1+alpha) - gammaln(alpha)-gammaln(delta1)   
    logRes = logC + (delta1-1)*np.log(x) - (delta1+alpha)*np.log(x+beta)
    return logRes

    
#### log likelyhood
def logLik(X:np.ndarray,data:np.ndarray):
    logP = 0
    delta1,alpha,beta = X
    for x in data:
        logP = logP + logPdfHyper(x,delta1,alpha,beta)
        
    return logP


def negLogLik(X:np.ndarray,data:np.ndarray):
    logP = 0
    delta1,alpha,beta = X
    for x in data:
        logP = logP + logPdfHyper(x,delta1,alpha,beta)
        
    return -logP


from scipy.optimize import minimize


def fitValues(data:np.ndarray):

    ## Valeurs initiales
    delta1 = 18
    alpha = 250
    beta = 11 #rate
    X0 = np.array([delta1,alpha,beta])

    dataFunc = np.array(data)
    bounds = [(10**-30, None), (10**-30, None), (10**-30, None)]

    res = minimize(negLogLik, X0,bounds=bounds, args=dataFunc,method='L-BFGS-B')
    
    if res.success:
        deltaFit,alphaFit,betaFit = res.x
        return deltaFit,alphaFit,betaFit
    
    else:
        raise Exception("Erreur convergence hyperparamètres")
        return 0  
    
    
from sklearn.metrics.cluster import pair_confusion_matrix

def getMetrics(Ztrue,Z):
    '''
    return [TPR,FDR]
    '''
    confM = pair_confusion_matrix(Ztrue,Z)
    TN,FP,FN,TP = confM.ravel()
    if (TP+FN)!= 0 :
        TPR = TP/(TP + FN)
    else : TPR = 0
    
    if (FP+TP) != 0: 
        FDR = FP/(FP + TP)
    else : FDR = 1
    ## Forme de la matrice 
    #  | TN  FP |
    #  | FN  TP |
    return  TPR,FDR