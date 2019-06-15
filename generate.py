import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Loading the data
with open('transforms.pkl', 'rb') as f:
	transforms = pickle.load(f)

with open('trajectory.pkl', 'rb') as f:
    trajectory = pickle.load(f)

with open('smoothTrajectory.pkl', 'rb') as f:
    smoothTrajectory = pickle.load(f)

difference = trajectory - smoothTrajectory

# Creating output video
v = cv2.VideoCapture(sys.argv[1])
W = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(v.get(cv2.CAP_PROP_FPS))
DispThresh = 50	
DispThresh += 10

curX = DispThresh
curY = DispThresh

out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (W-2*DispThresh, H-2*DispThresh))
count = 0 
m = np.zeros([2, 3])

while(v.isOpened()):
    ret, frame = v.read()
    if ret == True:
        if count > 0:
        	curX = DispThresh+difference[0][count-1]
        	curY = DispThresh+difference[1][count-1]
        
        # Writing to output file
        boxPart = frame[np.int(curY):np.int(curY)+H-2*DispThresh, np.int(curX):np.int(curX)+W-2*DispThresh, :]
        out.write(boxPart)
                
        # Displaying the difference
        cv2.rectangle(frame, (np.int(curX), np.int(curY)), (np.int(curX)+W-2*DispThresh, np.int(curY)+H-2*DispThresh), (0, 255, 0), 3)
        frame = cv2.resize(frame, (W-2*DispThresh, H-2*DispThresh))
        dispFrame = cv2.hconcat([frame, boxPart])
        if dispFrame.shape[1] > 1920:
        	dispFrame = cv2.resize(dispFrame, (dispFrame.shape[1]//2, dispFrame.shape[0]//2));
        cv2.imshow('DispFrame', dispFrame)
        
        count +=1
        if count == 1000:
            break

        if cv2.waitKey(20) & 0xFF == ord('q'):
        	break
    else:
        break

# When everything done, release the capture
v.release()
out.release()