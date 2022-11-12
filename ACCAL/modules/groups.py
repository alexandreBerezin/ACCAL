import numpy as np
from scipy.special import gammaln
from scipy.stats import truncnorm

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






def getNewSampleG(mean:tuple):

    m1,m2 = mean
    
    my_std = 0.1
    a1 = - m1 / my_std
    a2 = - m2 / my_std
    b = np.inf
    
    return (truncnorm.rvs(a1,b,loc=m1,scale=my_std),truncnorm.rvs(a2,b,loc=m2,scale=my_std))

def loglik(alpha:float,beta:float,data:np.ndarray):
    # return de log Likelyhood of the data
    N, = data.shape
    A = N*(alpha*np.log(beta) - gammaln(alpha))
    B = (alpha-1)* np.sum(np.log(data))
    C = -beta*np.sum(data)
     
    return A+B+C



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
    

