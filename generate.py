import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

def getAngle(a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

# Loading the data
with open('transforms.pkl', 'rb') as f:
	transforms = pickle.load(f)

with open('trajectory.pkl', 'rb') as f:
    trajectory = pickle.load(f)

with open('smoothTrajectory.pkl', 'rb') as f:
    smoothTrajectory = pickle.load(f)

# Box trajectory
difference = trajectory - smoothTrajectory

# Frame transform
smoothTransforms = transforms - difference

# Creating output video
v = cv2.VideoCapture(sys.argv[1])
W = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(v.get(cv2.CAP_PROP_FPS))
n_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
DispThresh = int(sys.argv[2])	
DispThresh += 10

OrigBox = [[DispThresh, W-DispThresh, W-DispThresh, DispThresh],
            [DispThresh, DispThresh, H-DispThresh, H-DispThresh],
            [1,           1,           1,             1        ]]

out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (W-2*DispThresh, H-2*DispThresh))
m = np.zeros([2, 3])

for i in range(n_frames-1):
    ret, frame = v.read()
    if ret == True:

        # Writing to output file
        m[0][0] = (smoothTransforms[3][i])*np.cos(smoothTransforms[2][i])
        m[0][1] = -(smoothTransforms[3][i])*np.sin(smoothTransforms[2][i])
        m[1][0] = (smoothTransforms[3][i])*np.sin(smoothTransforms[2][i])
        m[1][1] = (smoothTransforms[3][i])*np.cos(smoothTransforms[2][i])
        m[0][2] = smoothTransforms[0][i]
        m[1][2] = smoothTransforms[1][i]
        stable = cv2.warpAffine(frame, m, (W, H))
        boxPart = stable[DispThresh:H-DispThresh, DispThresh:W-DispThresh, :]
        out.write(boxPart)
                
        # Displaying the difference
        m[0][0] = (1+difference[3][i-1])*np.cos(difference[2][i-1])
        m[0][1] = -(1+difference[3][i-1])*np.sin(difference[2][i-1])
        m[1][0] = (1+difference[3][i-1])*np.sin(difference[2][i-1])
        m[1][1] = (1+difference[3][i-1])*np.cos(difference[2][i-1])
        m[0][2] = difference[0][i-1]
        m[1][2] = difference[1][i-1]
        recPts = np.matmul(m, OrigBox)
        
        pt0 = (int(round(recPts[0][0])), int(round(recPts[1][0])))
        pt1 = (int(round(recPts[0][1])), int(round(recPts[1][1])))
        pt2 = (int(round(recPts[0][2])), int(round(recPts[1][2])))
        pt3 = (int(round(recPts[0][3])), int(round(recPts[1][3])))
        
        white = (255, 255, 255)
        cv2.line(frame, pt0, pt1, white, 2)
        cv2.line(frame, pt1, pt2, white, 2)
        cv2.line(frame, pt2, pt3, white, 2)
        cv2.line(frame, pt3, pt0, white, 2)
        
        frame = cv2.resize(frame, (W-2*DispThresh, H-2*DispThresh))
        dispFrame = cv2.hconcat([frame, boxPart])
        
        # If output too wide
        if dispFrame.shape[1] >= 1920:
        	dispFrame = cv2.resize(dispFrame, (dispFrame.shape[1]//2, dispFrame.shape[0]//2));
        cv2.imshow('DispFrame', dispFrame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
        	break
    else:
        break

# When everything done, release the capture
v.release()
out.release()