from typing import Tuple
import numpy as np 

from scipy.stats import truncnorm


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



    
def fitValues(data:np.ndarray,alpha0:float,beta0:float,burnin:int,nbSample:int,deltaSample:int):
    """Fit values of the Distribution with a MCMC algorithm

    Args:
        data (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    
    AL,BL = mcmcSampling(data,alpha0,beta0,burnin,nbSample,deltaSample)
    
    delta = np.mean(AL)
    meanBL = np.mean(BL)
    varBL = np.std(BL)
    
    return delta,(meanBL**2)/varBL,meanBL/varBL 
    
    
def mcmcSampling(data:np.ndarray,alpha0:float,beta0:float,burnin:int,nbSample:int,deltaSample:int)->tuple:   
    """
    Gamma model 
    """
    
    xt = (alpha0,beta0)
    
    alphaL = []
    betaL =  []
    
    nbS = 0
    gap = 0
    
    ##### BURNIN STEP ####
            
    print("burnin step ...")
    
    for _ in range(burnin):
        print(_,"/",burnin,end='\r')
        
        #Curent state
        alpha,beta = xt
        
        #Sample new candidate
        xp = getNewSampleG(xt)
        alphap,betap = xp

        logA = loglik(alphap,betap,data) -loglik(alpha,beta,data)
        logB = getlogGPDF(xt,xp) - getlogGPDF(xp,xt)
        logP = np.min([0,logA+logB])
        
        #probability of acceptance
        if np.log(np.random.rand()) < logP :
            xt = xp
        
    
    
    ##### SAMPLING STEP ####
    print("Sampling ...")
    while nbS <= nbSample:
        
        #Curent state
        alpha,beta = xt
        
        #Sample new candidate
        xp = getNewSampleG(xt)
        alphap,betap = xp

        logA = loglik(alphap,betap,data) -loglik(alpha,beta,data)
        logB = getlogGPDF(xt,xp) - getlogGPDF(xp,xt)
        logP = np.min([0,logA+logB])
        
        #probability of acceptance
        if np.log(np.random.rand()) < logP :
            xt = xp

        gap = gap +1
        if gap == deltaSample:
            alphaL.append(xt[0])
            betaL.append(xt[1])
            gap = 0
            nbS = nbS +1
            print("\r",nbS,"/",nbSample,end='\r')
            

    return (alphaL,betaL)
        
    
    
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






from scipy.special import gammaln

def loglik(alpha:float,beta:float,data:np.ndarray):
    # return de log Likelyhood of the data
    N, = data.shape
    A = N*(alpha*np.log(beta) - gammaln(alpha))
    B = (alpha-1)* np.sum(np.log(data))
    C = -beta*np.sum(data)
     
    return A+B+C
        

def getProb(xp:tuple,xt:tuple,f:callable,g:callable)->float:
    alpha = f(xp)/f(xt)
    beta = g(xt,xp)/g(xp,xt)
    
    return np.min(1,alpha*beta)


def getNewSampleG(mean:tuple):

    m1,m2 = mean
    
    my_std = 0.1
    a1 = - m1 / my_std
    a2 = - m2 / my_std
    b = np.inf
    
    return (truncnorm.rvs(a1,b,loc=m1,scale=my_std),truncnorm.rvs(a2,b,loc=m2,scale=my_std))

def getlogGPDF(x,y):
    """
    return g(x1 |x2) * g(y1 |y2)
    """
    x1,x2 = x
    y1,y2 = y
    
    my_std = 0.1
    a1 = - y1 / my_std  
    a2 = - y2 / my_std
    
    b = np.inf
    
    res = truncnorm.logpdf(x1,a1,b,loc=y1,scale=my_std) + truncnorm.logpdf(x2,a2,b,loc=y2,scale=my_std)
    
    
    return res
    



            