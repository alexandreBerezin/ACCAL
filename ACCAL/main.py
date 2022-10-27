import pathlib
import functions
import visualisation

import subprocess
import modules.imageProcessing


######### Parameters ##########

appPath = pathlib.Path(r"D:\Stage\ACCAL\ACCAL") #### To change with the app folder
dataFolderPath = pathlib.Path(r"D:\Stage\ACCAL\data\dataTest2") #### To change with teh dataSet folder
featureAppPath = pathlib.Path(appPath,"features.py")


## Dissimilarity Matrix
DENOISE_RATIO = 0.1
CLIP_LIMIT = 0.01
LENGTH_KERNEL = 4.0
PIXEL_SIDE = 360
FEATURE_NUMBER = 250

NUMBER_CORES = 2 


## Clustering
DIST_CUT = 0.2
NB_BURNIN = 20
NB_SAMPLE = 50
EACH_SAMPLE = 2


## Visualisation
PROB_LIMIT = 0.5 


###################################################
###################################################


### Compute and save tomporary images used for further processing
#modules.imageProcessing.computeAndSaveTempImages(dataFolderPath=dataFolderPath,denoiseRatio=DENOISE_RATIO,clipLimit=CLIP_LIMIT)


### Compute and save features for each image 
#functions.saveFeatures(dataFolderPath=dataFolderPath,absAppPath=appPath,pixelSide=PIXEL_SIDE,lengthKernel=LENGTH_KERNEL,featureNumber=FEATURE_NUMBER)
#subprocess.run(["python",str(featureAppPath),str(dataFolderPath),str(appPath),str(PIXEL_SIDE),str(LENGTH_KERNEL),str(FEATURE_NUMBER),str(NUMBER_CORES)])

## Compute and save the dissimilarity Matrix
#functions.saveDissMatrix(dataFolderPath=dataFolderPath)


## Compute and save Probability matrix
functions.clustering(dataFolderPath=dataFolderPath,distCut=DIST_CUT,nbBurnin=NB_BURNIN,nbSample=NB_SAMPLE,eachSample=EACH_SAMPLE)


## Save Results
visualisation.saveResults(dataFolderPath=dataFolderPath,probLimit=PROB_LIMIT)

