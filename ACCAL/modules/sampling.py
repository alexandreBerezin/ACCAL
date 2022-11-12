import numpy as np 

from scipy.stats import truncnorm
from scipy.stats import beta


from scipy.special import gammaln




##########################################
###### Fonction pour sampling r,p ########

etha_ =1
sigma_ =1


def drawSampleMCMCforR(priorR,p,Z):
    '''
    Draw a new sample for r with Metropolis Hastings 
    '''
    
    r = priorR  
    #draw new candidate
    candidate = drawSampleTruncNormal(r)
    #logProbabiblty of acceptance
    logAlpha = logPosteriorR(candidate,p,Z)-logPosteriorR(r,p,Z)
    logBeta = logPdfTruncNormal(r,candidate)-logPdfTruncNormal(candidate,r)
    logProb = min(0,logAlpha+logBeta)
    #Accept new sample or keep the old one
    u = np.random.rand()
    if np.log(u)<logProb:
        #On accepte
        r = candidate

    return r


##### Auxiliary functions


def getNiK(Z):
    '''
    return N = [n0,n1,n2,...,n(K-1)]
    le nombre d'éléments de chaque cluster 
    ----------------
    return N,K
    '''
    
    N = []
    K = np.max(Z) + 1 #nombre de clusters
    Za = np.array(Z)
    for clustId in range(K):
        N.append(np.sum(Za==clustId))
    return N,K
        
    
def logPosteriorR(r:float,p:float,Z:list)->float:
    '''
    Calcule le log de la densité (proportionnelle) de r|p,rho
    '''
    ######
    ## etha -> shape 
    ## sigma -> rate
    ## rate = 1/scale /!\
    
    N,K = getNiK(Z)
    
    A = (etha_-1)*np.log(r)
    B = K * ( r*np.log(p) -gammaln(r)) 
    C = -r*sigma_
    
    D = 0 
    
    for j in range(K):
        nj = N[j]
        D = D+ gammaln(nj-1+r)
        
    return A+B+C+D


def logPdfTruncNormal(x,mu):
    
    my_std = 0.1
    
    a = -mu/my_std
    b = np.inf
    return truncnorm.logpdf(x,a,b,loc=mu,scale=my_std)


def drawSampleTruncNormal(mu:float):
    '''
    Draw a sample from a truncated normal distribution
    with mean as mu
    '''
    my_std = 0.1
    
    a = -mu/my_std
    b = np.inf

    return truncnorm.rvs(a,b,loc=mu,scale=my_std)





##########################################
####### Fonctions pour sampling p ########

u_ = 1
v_ = 1



def drawSampleforP(r,Z,n):
    _,K=getNiK(Z)

    return beta(n-K+u_,r*K+v_).rvs()


##########################################
####### Fonctions pour sampling Z ########


def getNewSampleZforI(D:np.ndarray,Za:np.ndarray,i:int,p:float,r:float,
                    alpha:float,beta:float,delta1:float,
                    zeta:float,gamma_:float,delta2:float)->np.ndarray:
    '''
    Get a new sample for Z[i] according the the parameters
    Z : np.ndarray representing cluster allocation 
    i : index of the element that is going to be clustered Zi 
    
    return new array Z
    
    '''
    

    
    Za = np.array(Za)
    
    maxZaInit = np.max(Za)
    
    
    alone = True ## Bool representing if element i is alone in his cluster
    nbSameCluster = np.sum(Za==Za[i]) #number of element in the same cluster 
    if nbSameCluster > 1:
        alone = False
        

    idClustInit = Za[i] # index of the initial cluster i is in 
    Za[i]=-1 ## To indicate that the element in index i has no longer a cluster
    
    
    logProbVec = [] # initial vector for log probability 
    
    if alone == False:
        # for all clusters k there is at least a coin 
        Kmi = np.max(Za) + 1 # number of cluster without element i 
        idNewCluster = np.max(Za) + 1 # the index of the new cluster if choosen
        
        for k in range(np.max(Za) + 1):
            ## for all clusters from 0 to np.max(Za)
            logP = getLogProbClustZiInK_notEmpty(D=D,Za=Za,i=i,k=k,
                                                 p=p, r=r,
                                                 alpha=alpha,beta=beta,delta1=delta1,
                                                 zeta=zeta,gamma_=gamma_,delta2=delta2)
            logProbVec.append(logP)
          
          
        ##### Append prob for New cluster  
        logLik2 = 0 
        for t in range(np.max(Za)+1): 
            # for all clusters 
            ntmi = np.sum(Za==Za[t])
            zetaIT = zeta + delta2*ntmi
            gammaIT = gamma_ + np.sum( [ D[i,j] for j in np.argwhere(Za==t).ravel() ]   )
            
            logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma_) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
                + np.sum(  [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in np.argwhere(Za==t).ravel()]   )
    
        
        logP = np.log(Kmi +1) + r*np.log(1-p) + logLik2     
        logProbVec.append(logP)

    else:
        ## single element in a cluster  
        idEmptyCluster = idClustInit
        
        Kmi = np.max(Za)  # number of clusters without i 
        
        for k in range(np.max(Za) + 1):
            if k==idEmptyCluster:
                logLik2 = 0 
                for t in range(Kmi+1):
                    if t!= idEmptyCluster:
                        ntmi = np.sum(Za==Za[t])
                        zetaIT = zeta + delta2*ntmi
                        gammaIT = gamma_ + np.sum( [ D[i,j] for j in np.argwhere(Za==t).ravel() ]   )
                        
                        logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma_) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
                            + np.sum(  [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in np.argwhere(Za==t).ravel()]   )
            
                logP = np.log(Kmi +1 ) + r*np.log(1-p) + logLik2     
                logProbVec.append(logP)
            else:
                logP = getLogProbClustZiInK_Empty(D=D,Za=Za,i=i,k=k,
                                                  p=p,r=r,
                                                  alpha=alpha,beta=beta,delta1=delta1,
                                                  zeta=zeta,gamma_=gamma_,delta2=delta2,
                                                  idEmptyCluster=idEmptyCluster,maxZaInit = maxZaInit)
                logProbVec.append(logP)
                
        
    #### choose new cluster 
    
    logProbVec = np.array(logProbVec)
    maxLogProb = np.max(logProbVec)
        
    logProbVec = logProbVec - maxLogProb
    propToP = np.exp(logProbVec)
    pVec = propToP/np.sum(propToP)
    
    #### if new cluster is not the old one 
    #### move all id above new cluster to minus 1 
    
    s, = np.shape(pVec)
    #print(f"shape vecProb{s}, max Za init Za : {maxZaInit}")
    
    newK = np.random.choice(np.arange(s),p=pVec)

    Za[i] = newK
    if (alone == True and newK!=idClustInit):
        ### it means that the old cluster does not exists
        ### So we move an id down every clusters avec idClust Init
        Za[Za>idClustInit] += -1 
        
    return  Za


def getLogProbClustZiInK_notEmpty(D:np.ndarray,Za:np.ndarray,i:int,k:int,
                                  p:float,r:float,
                                  alpha:float,beta:float,delta1:float,
                                  zeta:float,gamma_:float,delta2:float):
    '''
    Return the log Probability that Zi = k 
    
    OOOKKKK
    '''
    nkmi = np.sum(Za==Za[k]) ## Number of element in cluster k
    if nkmi == 0:
        raise Exception("Error in an argument : nkmi should be not equal to 0")
    
    ### calcul logLik1
    alphaIK = alpha + delta1*nkmi
    betaIK = beta + np.sum( [ D[i,j] for j in np.argwhere(Za==k).ravel() ]   )
    
    logLik1 = gammaln(alphaIK) + alpha*np.log(beta) -gammaln(alpha) \
        - alphaIK*np.log(betaIK) \
            + np.sum(  [(delta1-1)*np.log(D[i,j]) -gammaln(delta1) for j in np.argwhere(Za==k).ravel()] )
            
    ### Calcul logLik2 
    Kmi = np.max(Za)+1 ## number of clusters without i 

    logLik2 = 0 
    for t in range(Kmi+1):
        # for all clusters in Z
        if t != k :
            ntmi = np.sum(Za==Za[t])
            zetaIT = zeta + delta2*ntmi
            gammaIT = gamma_ + np.sum( [ D[i,j] for j in np.argwhere(Za==t).ravel() ]   )
            
            logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma_) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
                + np.sum(  [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in np.argwhere(Za==t).ravel()]   )
    
    ### logProbability 
    logP = np.log(nkmi +1) + np.log(p) + np.log(nkmi -1 +r) + logLik1 + logLik2 - np.log(nkmi) 
    return logP


def getLogProbClustZiInK_Empty(D:np.ndarray,Za:np.ndarray,i:int,k:int,
                               p:float,r:float,
                               alpha:float,beta:float,delta1:float,
                               zeta:float,gamma_:float,delta2:float,
                               idEmptyCluster:int,maxZaInit:int):
    '''
    Return the log Probability that a coin with index i,
    in in the cluster k. 
    Assuming that the empty cluster has in id idEmptyCluster 
    
    
    OOOK ? 
    '''
    nkmi = np.sum(Za==Za[k])
    if nkmi == 0:
        raise Exception("Error in an argument : nkmi should be not equal to 0")
    
    ### calcul logLik1
    alphaIK = alpha + delta1*nkmi
    betaIK = beta + np.sum( [ D[i,j] for j in np.argwhere(Za==k).ravel() ]   )
    
    logLik1 = gammaln(alphaIK) + alpha*np.log(beta) -gammaln(alpha) \
        - alphaIK*np.log(betaIK) \
            + np.sum(  [(delta1-1)*np.log(D[i,j]) -gammaln(delta1) for j in np.argwhere(Za==k).ravel()] )
            
    ### Calcul logLik2 
    logLik2 = 0 
    for t in range(maxZaInit +2  ):
        ntmi = np.sum(Za==Za[t])
        if t != k and t!=idEmptyCluster and ntmi!= 0 :
            zetaIT = zeta + delta2*ntmi
            gammaIT = gamma_ + np.sum( [ D[i,j] for j in np.argwhere(Za==t).ravel() ]   )
            
            logLik2 = logLik2 + gammaln(zetaIT) + zeta*np.log(gamma_) - gammaln(zeta) -zetaIT*np.log(gammaIT) \
                + np.sum(  [(delta2-1)*np.log(D[i,j]) -gammaln(delta2) for j in np.argwhere(Za==t).ravel()]   )
    
    ### logProbability 
    logP = np.log(nkmi +1) + np.log(p) + np.log(nkmi -1 +r) + logLik1 + logLik2 - np.log(nkmi) 
    return logP



#############################################
#############################################



def getProbFinalij(i:int,j:int,samples:np.ndarray):
    s = 0
    M,_ = np.shape(samples)
    for c in samples:
        if c[i] == c[j]:
            s = s+1
    return s/M


def getProbMatrix(samples):
    _,n =np.shape(samples)
    
    mat = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i):
            mat[i,j] = getProbFinalij(i,j,samples)
            
    mat = mat + np.transpose(mat)
    np.fill_diagonal(mat,1)
    
    return mat

            
            
        


