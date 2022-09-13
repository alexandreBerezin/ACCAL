import subprocess
import pathlib

appPath = pathlib.Path(r"D:\Stage\ACCAL\ACCAL") #### To change with the app folder
dataFolderPath = pathlib.Path(r"D:\Stage\ACCAL\data\dataTest1") #### To change with teh dataSet folder


featureAppPath = pathlib.Path(appPath,"features.py")

## Dissimilarity Matrix
DENOISE_RATIO = 0.1
CLIP_LIMIT = 0.01
LENGTH_KERNEL = 4.0
PIXEL_SIDE = 360
FEATURE_NUMBER = 250


## Clustering
DIST_CUT = 0.2
NB_BURNIN = 20
NB_SAMPLE = 50
EACH_SAMPLE = 2


## Visualisation
PROB_LIMIT = 0.5 


#_,dataFolderPath,absAppPath,pixelSide,lengthKernel,featureNumber = sys.argv

print("START MAIN Program")


subprocess.run(["python",str(featureAppPath),str(dataFolderPath),str(appPath),str(PIXEL_SIDE),str(LENGTH_KERNEL),str(FEATURE_NUMBER)])