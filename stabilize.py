import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp

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
fx = cp.Variable(len(trajectory[0]))
fy = cp.Variable(len(trajectory[1]))

lbd1 = 1000
lbd2 = 100
lbd3 = 10000
DispThresh = 50
constraints = [cp.abs(fx - trajectory[0]) <= DispThresh,
 			   cp.abs(fy - trajectory[1]) <= DispThresh]

# Defining the minimization objective function
obj = 0																																																																																								
for i in range(len(trajectory[0])):
	obj += ( (trajectory[0][i]-fx[i])**2 + (trajectory[1][i]-fy[i])**2 )

# DP1
for i in range(len(trajectory[0])-1):
	obj += lbd1*(cp.abs(fx[i+1]-fx[i]) + cp.abs(fy[i+1]-fy[i]))

# DP2
for i in range(len(trajectory[0])-2):
	obj += lbd2*(cp.abs(fx[i+2]-2*fx[i+1]+fx[i]) + cp.abs(fy[i+2]-2*fy[i+1]+fy[i]))

# DP3
for i in range(len(trajectory[0])-3):
	obj += lbd3*(cp.abs(fx[i+3]-3*fx[i+2] + 3*fx[i+1]-fx[i]) + cp.abs(fy[i+3]-3*fy[i+2]+3*fy[i+1]-fy[i]))

prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()

# Results
plt.subplot(1, 2, 1)
plt.plot(trajectory[0])
plt.plot(fx.value)
plt.subplot(1, 2, 2)
plt.plot(trajectory[1])
plt.plot(fy.value)
plt.show()

smoothTrajectory = np.array([fx.value, fy.value])

# Storing smooth trajectory
with open('smoothTrajectory.pkl', 'wb') as f:
	pickle.dump(smoothTrajectory, f)
