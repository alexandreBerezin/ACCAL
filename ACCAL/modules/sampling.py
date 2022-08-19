import numpy as np 

from scipy.stats import truncnorm
from scipy.stats import beta


from scipy.special import gamma,gammaln


def getRandomZ(r:float,p:float,n:int)->np.ndarray:
    '''
    Create a random vector Z from the microcluster model
    with cluster sizes ~ negBin(r,p) + 1
    
    ----------
    output
    Z = [ 3,2,2,43,0,0, .... , 23]
    '''
    
    #Reject sampling
    clusterSize = []
    somme = 0
    while(somme!= n):
        S = np.random.negative_binomial(r, p) +1
        clusterSize.append(S)
        somme = somme + S
        if somme > n:
            clusterSize = []
            somme = 0 

    
    # Random permutation 
    vec =[]
    for i in range(len(clusterSize)):
        size = clusterSize[i]
        for j in range(size):
            vec.append(i)
    Z = np.random.permutation(vec)
    return Z 



##########################################
###### Fonction pour sampling r,p ########

etha_ =1
sigma_ =1


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
    B = K * ( r*np.log(p) -np.log(gamma(r))) 
    C = -r*sigma_
    
    D = 0 
    
    for j in range(K):
        nj = N[j]
        D = D+ gammaln(nj-1+r)
        
    return A+B+C+D


def PdfTruncNormal(x,mu):
    lower, upper = 0, np.inf
    sig = 1
    dist = truncnorm(
        (lower - mu) / sig, (upper - mu) / sig, loc=mu, scale=sig)
    
    return dist.pdf(x)


def drawSampleTruncNormal(mu):
    '''
    Draw a sample from a truncated normal distribution
    with mean as mu
    '''
    lower, upper = 0, np.inf
    sig = 1
    dist = truncnorm(
        (lower - mu) / sig, (upper - mu) / sig, loc=mu, scale=sig)
    return dist.rvs()


def drawSampleMCMCforR(priorR,p,Z):
    '''
    Draw a new sample for r with Metropolis Hastings 
    '''
    
    r = priorR  
    #draw new candidate
    candidate = drawSampleTruncNormal(r)
    #logProbabiblty of acceptance
    logAlpha = logPosteriorR(candidate,p,Z)-logPosteriorR(r,p,Z)
    logBeta = np.log(PdfTruncNormal(r,candidate))-np.log(PdfTruncNormal(candidate,r))
    logProb = min(0,logAlpha+logBeta)
    #Accept new sample or keep the old one
    u = np.random.rand()
    if np.log(u)<logProb:
        #On accepte
        r = candidate

    return r


##########################################
####### Fonctions pour sampling p ########

u_ = 1
v_ = 1



def drawSampleforP(r,Z,n):
    _,K=getNiK(Z)
    frBeta = beta(r*K+v_,n-K+u_)
    return frBeta.rvs()


##########################################
####### Fonctions pour sampling Z ########



### Param 

delta1 = 2
alpha_ = 2
beta_ = 1


delta2 = 7
xsi_ = 2
gamma_ = 1


def getNiRho(rho:list)->list:
    '''
    Renvoie un vecteur des nombre des éléments de rho
    '''
    N = []
    for clst in rho:
        N.append(len(clst))
    
    return N


def getZ(rho:list,n:int)->np.ndarray: 
    '''
    Compute Z from rho
    '''
    Znew = np.zeros(n,dtype = int)
    for idClust,clust in enumerate(rho):
        for clustId in clust:
            Znew[clustId] = idClust
            
    return Znew

def getRho(Z:np.ndarray)->list:
    '''
    Compute rho from Z
    '''
    K = np.max(Z)+1
    rho = []
    for idClust in range(K):
        arg = list(np.argwhere(Z==idClust).ravel())
        if arg!= []:
            rho.append(arg)
    return rho


def getLik1(rhomi:list,i:int,k:int,D:np.ndarray)->float:
    '''
    Calcul intérmédiaire de Lik1
    '''
    
    nmi = getNiRho(rhomi)
    
    ### Calucul alpha_ik
    alphaik= alpha_ + delta1*nmi[k]
    
    ##Calcul Beta ik 
    clustk = rhomi[k]
    s = 0 
    for j in clustk:
        s = s + D[i,j]
    betaik = beta_ + s

    #### Caclul Lik1
    #### Lik1 = [ Gamma(alphaik)/Gamma(alpha) ] * [ Beta^alpha/ beta_ik ^alphaik ] * P 
    
    
    #A = gamma(alphaik)/gamma(alpha_)
    
    #A = np.exp(gammaln(alphaik) - gammaln(alpha_))
    
    logA = gammaln(alphaik) - gammaln(alpha_)
    
    #B = beta_**(alpha_)/(betaik**(alphaik))
    
    logB = alpha_*np.log(beta_) - alphaik*np.log(betaik)
    #B = np.exp(logB)
    
    
    #PI = 1
    #for j in clustk:
    #    PI=PI*(D[i,j]**(delta1-1))/(gamma(delta1))

    
    logPI = 0
    for j in clustk:
        logPI=logPI + (delta1-1)*np.log(D[i,j]) - gammaln(delta1)

        
    return np.exp(logA+logB+logPI)


def getLik2(rhomi:list,i:int,k:int,D:np.ndarray)->float:
    '''
    Calcul intérmédiaire de Lik2
    '''
    nmi = getNiRho(rhomi)
    Kmi = len(nmi)
    
    pTot = 10**100
    for t in range(Kmi):
        if t!=k:
            ##Calcul Xsiit
            ksiit=xsi_ + delta2*nmi[t]
            
            
            ##Calcul Gammait
            clustt = rhomi[t]
            s= 0 
            for j in clustt:
                s = s + D[i,j]
        
            gammait = gamma_ + s
            
            ## Calcul de Lik2
            #A = gamma(ksiit)/(gamma(xsi_))
            #B = gamma_**(xsi_)/gammait**(ksiit)
            
            
            logAB = gammaln(ksiit) + xsi_*np.log(gamma_) - gammaln(xsi_) - ksiit*np.log(gammait)
            #logAB = np.log(gamma(ksiit)) + xsi_*np.log(gamma_) - np.log(gamma(xsi_)) - ksiit*np.log(gammait)
            
            
            logPI = 0 #produit
            for j in clustt:
                logPI =logPI + (delta2-1)*np.log(D[i,j]) - gammaln(delta2)
                
            #AB = np.exp(Alog+Blog)
            
            logInter = logAB + logPI
            
            pTot = pTot * np.exp(logInter) 
            
            #pTot=pTot*A*B*PI

    
    return pTot


def getProbZiClustk(rhomi:list,i:int,k:int,p:float,r:float,D)->float:
    '''
    Calcul de la probabilité d'avoir l'élément i appartenant au cluster
    k de rhomi
    '''
    
    nmi = getNiRho(rhomi)
    
    Kmi= len(rhomi)
    
    
    if k<=Kmi-1:
        Lik1 = getLik1(rhomi,i,k,D)
        Lik2 = getLik2(rhomi,i,k,D)
                
        nkmi = nmi[k]
        
        prob = (nkmi+1)*(1-p)*(nkmi-1+r)*Lik1*Lik2/nkmi
        
        assert(prob!= np.nan)
        
        return prob
    
    ##Nouveau cluster
    if k == Kmi:
        Lik2 = getLik2(rhomi,i,k,D)
        prob = (Kmi+1)*(p)**r*Lik2
        return prob
    
    else:
        print("ERREUR")
        
        
def sampleNewZForI(i:int,p:float,r:float,Z:np.ndarray,D:np.ndarray)->np.ndarray:
    '''
    Calcul un nouveau sample Z en samplant un nouveau Z_i 
    '''
    
    #nombre d'éléments
    n, = np.shape(Z)
    ### transform Z
    Z[i] = -1
    rhomi = getRho(Z)
    
    
    Kmi = len(rhomi)
    
    probVec = np.zeros(Kmi+1)
    
    for k in range(Kmi+1):
        probVec[k] = getProbZiClustk(rhomi,i,k,p,r,D)
        
    #print(probVec)
    probVec = probVec/np.sum(probVec)
    
    
    #print(probVec)

    
    newK = np.random.choice(range(Kmi+1),p=probVec)
    
    if newK == Kmi:
        #Nouveau cluster
        rhomi.append([i])
    else:
        rhomi[newK].append(i)
        
    return getZ(rhomi,n)


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

            
            
        




