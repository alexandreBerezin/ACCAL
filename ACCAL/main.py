import pathlib
import functions

import modules.imageProcessing

######### Parameters ##########

appPath = pathlib.Path(r"D:\Stage\ACCAL\ACCAL") #### To change with the app folder
dataFolderPath = pathlib.Path(r"D:\Stage\ACCAL\data\dataTest1") #### To change with teh dataSet folder
DENOISE_RATIO = 0.1
CLIP_LIMIT = 0.01
LENGTH_KERNEL = 4.0
PIXEL_SIDE = 360

FEATURE_NUMBER = 250

DIST_CUT = 0.2
NB_BURNIN = 20
NB_SAMPLE = 50
EACH_SAMPLE = 2



### Compute and save 


### Compute and save tomporary images used for further processing
#modules.imageProcessing.computeAndSaveTempImages(dataFolderPath=dataFolderPath,denoiseRatio=DENOISE_RATIO,clipLimit=CLIP_LIMIT)


### Compute and save features of each images according to 
#functions.saveFeatures(dataFolderPath=dataFolderPath,absAppPath=appPath,pixelSide=PIXEL_SIDE,lengthKernel=LENGTH_KERNEL,featureNumber=FEATURE_NUMBER)


#functions.saveDissMatrix(dataFolderPath=dataFolderPath)

### Clustering

functions.clustering(dataFolderPath=dataFolderPath,distCut=DIST_CUT,nbBurnin=NB_BURNIN,nbSample=NB_SAMPLE,eachSample=EACH_SAMPLE)


