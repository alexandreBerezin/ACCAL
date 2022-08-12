from multiprocessing import Pool
import time

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm


# TO CHANGE
# absolute path to ACCAL/ACCAL folder
appPath = Path(r"D:\Stage\ACCAL\ACCAL")
sys.path.append(str(appPath))

#Kernel module
import numpy as np
import modules.features.kernel
import modules.features.selection
import scipy.sparse


# Path to data 
dataFolderPath = Path(r"D:\Stage\ACCAL\data\dataTest1","temp","processedImages")


K = modules.features.kernel.getK(l=4.0,absAppPath=appPath ,pixelSide=360)
pathList = sorted(list(dataFolderPath.glob("*.npy")))


print(__name__)



#Pool of 4 workers (4 cores)
if __name__ == "__main__": 
    t1 = time.perf_counter()
    pool = Pool(4)
    processes = [pool.apply_async(modules.features.selection.getFeatures, args=(250,K,path)) for path in pathList]
    tf = time.perf_counter()

    result = [p.get() for p in processes]
    dt = tf-t1
    print(dt)
    pool.close()


