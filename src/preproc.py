import cv2
import numpy as np
import pickle
import sys

sift = cv2.xfeatures2d.SIFT_create(200)

def getAffMat(I1, I2):
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Finding sift features
    kp1, desc1 = sift.detectAndCompute(I1, None)
    kp2, desc2 = sift.detectAndCompute(I2, None)

    # Finding good matches using ratio testing
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    pts_src = []
    pts_dst = []
    for i in range(len(good)):
    	pts_src.append([kp1[good[i].queryIdx].pt[0], kp1[good[i].queryIdx].pt[1]])
    	pts_dst.append([kp2[good[i].trainIdx].pt[0], kp2[good[i].trainIdx].pt[1]])

    pts_src = np.array(pts_src).astype(np.float32)
    pts_dst = np.array(pts_dst).astype(np.float32)

    # Computing affine matrix using the best matches
    return cv2.estimateAffinePartial2D(pts_src, pts_dst)[0]

v = cv2.VideoCapture(sys.argv[1])
n_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

# Generating the Xdata and Ydata
transforms = [[], [], [], []]
count = 0
while v.isOpened():
    
    ret, frame = v.read()

    if ret == True:

        if count > 0:
            transMat = getAffMat(prev, frame)
            try:
                transforms[0].append(transMat[0][2])
                transforms[1].append(transMat[1][2])
                transforms[2].append(np.arctan2(transMat[1][0], transMat[0][0]))
                transforms[3].append(np.sqrt(transMat[1][0]**2 + transMat[0][0]**2))
            except:
            	transforms[0].append(0)
            	transforms[1].append(0)
            	transforms[2].append(0)
            	transforms[3].append(1)
            
        count += 1
        prev = frame
        print( str((count/n_frames)*100) + "% completed")
    else:
    	break

v.release()

# Storing the data
with open('transforms.pkl', 'wb') as f:
	pickle.dump(transforms, f)