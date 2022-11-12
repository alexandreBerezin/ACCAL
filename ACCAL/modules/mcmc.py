############# For r #################
from scipy.special import gammaln
import numpy as np

from scipy.stats import truncnorm
from scipy.stats import beta


def drawSampleMCMCforR(priorR:float,p:float,rho:list,etha:float,sigma:float):
    '''
    Draw a new sample for r with Metropolis Hastings 
    '''
    
    r = priorR  
    #draw new candidate
    candidate = _drawSampleTruncNormal(r)
    #logProbabiblty of acceptance
    logAlpha = _logPosteriorR(candidate,p,rho,etha,sigma)-_logPosteriorR(r,p,rho,etha,sigma)
    logBeta = _logPdfTruncNormal(r,candidate)-_logPdfTruncNormal(candidate,r)
    logProb = min(0,logAlpha+logBeta)
    #Accept new sample or keep the old one
    u = np.random.rand()
    if np.log(u)<logProb:
        #On accepte
        r = candidate
    return r



def drawSampleforP(r:float,rho:list,N:int,u:float,v:float):
    '''
    n : number of element
    '''
    
    K=len(rho)
    return beta(N-K+u,r*K+v).rvs()


def drawSampleRhoI(rho:list,D:np.ndarray,Z:np.ndarray,i:int,p:float,r:float,
                     delta1:float,alpha:float,beta:float,
                     delta2:float,zeta:float,gamma:float):

    idClustInit = Z[i]

    rho[idClustInit].remove(i)

    if rho[idClustInit] == set():
        rho.remove(set())
        Z[Z>idClustInit] += -1

    logPList = []

    for k in range(len(rho)):
        logP = _getLogPZiK(rho,D,i,k,p,r,delta1,alpha,beta,delta2,zeta,gamma)
        logPList.append(logP)

    logP= _getLogPNewClust(rho,D,i,p,r,delta2,zeta,gamma)
    logPList.append(logP)


    logPVec = np.array(logPList)
    
    
    pVec = np.exp(logPVec)
    pVec = pVec / np.sum(pVec)
    
    l,=pVec.shape
    newK = np.random.choice(np.arange(l),p=pVec)
    
    

    if newK != l-1:
        rho[newK].add(i)
        Z[i]=newK
    else:
        rho.append({i})
        Z[i]=newK
    
    
    return rho,Z










##################################
###########AUX FUNCTIONS##########

def _getLogPZiK(rho:list,D:np.ndarray,i:int,k:int,p:float,r:float,
              delta1:float,alpha:float,beta:float,
              delta2:float,zeta:float,gamma:float):
    
    nkmi = len(rho[k])
    
    ### calcul logLik1
    alphaIK = alpha + delta1*nkmi
    betaIK = beta + sum( [ D[i,j] for j in rho[k] ] )
    
    logLik1 = gammaln(alphaIK) + alpha*np.log(beta) -gammaln(alpha) \
        - alphaIK*np.log(betaIK) \
            + sum(  [(delta1-1)*np.log(D[i,j]) -gammaln(delta1) for j in rho[k]] )
    
    #calcul logLik2
    logLik2 = 0 
    for t,clusterT in enumerate(rho):
        # for all clusters in Z
        if t != k :
            ntmi = len(clusterT)
            zetaIT = zeta + delta2*ntmi
            gammaIT = gamma + sum( [ D[i,j] for j in rho[t] ])

            logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
                + sum( [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in clusterT ] )

    ### logProbability 
    logP = np.log(nkmi +1) + np.log(p) + np.log(nkmi -1 +r) + logLik1 + logLik2 - np.log(nkmi) 
    return logP

def _getLogPNewClust(rho:list,D:np.ndarray,i:int,p:float,r:float,
                  delta2:float,zeta:float,gamma:float):
    ## Calcul logLik2 
    logLik2 = 0 
    for t,clusterT in enumerate(rho):
        ntmi = len(clusterT)
        zetaIT = zeta + delta2*ntmi
        gammaIT = gamma + sum( [ D[i,j] for j in clusterT ])

        logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
            + sum( [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in clusterT ] )

    logP = np.log(len(rho) +1) + r*np.log(1-p) + logLik2     
    return logP

    
def _logPosteriorR(r:float,p:float,rho:list,etha:float,sigma:float)->float:
    '''
    Calcule le log de la densité (proportionnelle) de r|p,rho
    '''
    K = len(rho)
    
    A = (etha-1)*np.log(r)
    B = K * ( r*np.log(1-p) -gammaln(r)) 
    C = -r*sigma
    
    D = 0  
    for clust in rho:
        D = D + gammaln(len(clust)-1+r)
        
    return A+B+C+D



def _logPdfTruncNormal(x,mu):
    
    my_std = 0.1
    
    a = -mu/my_std
    b = np.inf
    return truncnorm.logpdf(x,a,b,loc=mu,scale=my_std)


def _drawSampleTruncNormal(mu:float):
    '''
    Draw a sample from a truncated normal distribution
    with mean as mu
    '''
    my_std = 0.1
    
    a = -mu/my_std
    b = np.inf

    return truncnorm.rvs(a,b,loc=mu,scale=my_std)

