
import cv2
import numpy as np
import pathlib

from scipy.spatial import procrustes

def getFilteredMatch(img1:np.ndarray,features1:np.ndarray,img2:np.ndarray,features2:np.ndarray,reprojThreshold:float)->list:
    """return filtered Keypoint and  match between them

    Args:
        img1 (np.ndarray): _description_
        features1 (np.ndarray): _description_
        img2 (np.ndarray): _description_
        features2 (np.ndarray): _description_
        reprojThreshold (float): _description_

    Returns:
        list: _description_
    """
    keyPoint1 = []

    for coord in features1:
        x,y = coord
        keyPoint1.append(cv2.KeyPoint(int(x),int(y),1))
        
    keyPoint2 = []

    for coord in features2:
        x,y = coord
        keyPoint2.append(cv2.KeyPoint(int(x),int(y),1))
        
    
    orb = cv2.ORB_create()

    keyPoint1,des1 = orb.compute(img1,keyPoint1)
    keyPoint2,des2 = orb.compute(img2,keyPoint2)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    
    matches = sorted(matches,key=lambda x:x.distance)
    
    src_pts = np.float32([ keyPoint1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keyPoint2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,reprojThreshold)

    MatchesF = np.extract(mask.ravel(),matches)
    
    return [keyPoint1,keyPoint2,MatchesF]

def getNbAndP(keyPoint1,keyPoint2,match):
    src_pts = np.float32([ keyPoint1[m.queryIdx].pt for m in match ]).reshape(-1,2)
    dst_pts = np.float32([ keyPoint2[m.trainIdx].pt for m in match ]).reshape(-1,2)
    
    a,b,p=procrustes(src_pts,dst_pts)
    
    nb = np.shape(match)[0]
    
    return [nb,p]
    
    
def getImgFeat(tempPath:pathlib.Path,name:str)->tuple:
    """load img anf feature coordonates

    Args:
        tempPath (pathlib.Path): Path to temp folder
        name (str): name of the image (without extention) 

    Returns:
        tuple: returns [img,coord]
    """

    img = cv2.imread(str(pathlib.Path(tempPath,"croppedImages",name+".png")),cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nbPixel,_,_ = np.shape(img)
    feat = np.load(pathlib.Path(tempPath,"features",name+".npy"))
    coord = np.array([[i%nbPixel,i//nbPixel] for i in feat])
    return img,coord

