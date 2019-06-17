import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
import sys

# Loading the data
with open('transforms.pkl', 'rb') as f:
	transforms = pickle.load(f)

# Computing the trajectory
trajectory = np.cumsum(transforms, axis=1)
with open('trajectory.pkl', 'wb') as f:
	pickle.dump(trajectory, f)

# Smoothening the trajectories
# fx is the optimal x trajectory
# fy is the optimal y trajectory
# fth is the optimal theta trajectory
# fs is the optimal scale trajectory
fx = cp.Variable(len(trajectory[0]))
fy = cp.Variable(len(trajectory[1]))
fth = cp.Variable(len(trajectory[2]))
fs = cp.Variable(len(trajectory[3]))

lbd1 = 10000
lbd2 = 1000
lbd3 = 100000
DispThresh = int(sys.argv[1])
constraints = [cp.abs(fx - trajectory[0]) <= DispThresh,
 			   cp.abs(fy - trajectory[1]) <= DispThresh,
 			   cp.abs(fth - trajectory[2]) <= 0.05,
 			   cp.abs(fs - trajectory[3]) <= 0.01]

# Defining the minimization objective function
obj = 0																																																																																								
for i in range(len(trajectory[0])):
	obj += ( (trajectory[0][i]-fx[i])**2 + (trajectory[1][i]-fy[i])**2 + (trajectory[2][i]-fth[i])**2 + (trajectory[3][i]-fs[i])**2 )

# DP1
for i in range(len(trajectory[0])-1):
	obj += lbd1*(cp.abs(fx[i+1]-fx[i]) + cp.abs(fy[i+1]-fy[i]) + cp.abs(fth[i+1]-fth[i]) + cp.abs(fs[i+1]-fs[i]))

# DP2
for i in range(len(trajectory[0])-2):
	obj += lbd2*(cp.abs(fx[i+2]-2*fx[i+1]+fx[i]) + cp.abs(fy[i+2]-2*fy[i+1]+fy[i]) + cp.abs(fth[i+2]-2*fth[i+1]+fth[i]) + cp.abs(fs[i+2]-2*fs[i+1]+fs[i]))

# DP3
for i in range(len(trajectory[0])-3):
	obj += lbd3*(cp.abs(fx[i+3]-3*fx[i+2] + 3*fx[i+1]-fx[i]) + cp.abs(fy[i+3]-3*fy[i+2]+3*fy[i+1]-fy[i]) + cp.abs(fth[i+3]-3*fth[i+2]+3*fth[i+1]-fth[i]) + cp.abs(fs[i+3]-3*fs[i+2]+3*fs[i+1]-fs[i]))

prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()

# Results
plt.subplot(2, 2, 1)
plt.plot(trajectory[0])
plt.plot(fx.value)
plt.subplot(2, 2, 2)
plt.plot(trajectory[1])
plt.plot(fy.value)
plt.subplot(2, 2, 3)
plt.plot(trajectory[2])
plt.plot(fth.value)
plt.subplot(2, 2, 4)
plt.plot(trajectory[3])
plt.plot(fs.value)
plt.show()

smoothTrajectory = np.array([fx.value, fy.value, fth.value, fs.value])

# Storing smooth trajectory
with open('smoothTrajectory.pkl', 'wb') as f:
	pickle.dump(smoothTrajectory, f)
