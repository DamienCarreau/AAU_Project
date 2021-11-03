import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


# generate function f(x, y, z) = (2x+y*z**2, y/sqrt(x+1))
x = y = z = np.linspace(0,10,30)
X = np.array([[i, j, k] for i in x for j in y for k in z]).reshape(-1, 3, 1)
f = [[2*i+j*k**2, j/np.sqrt(i+1)] for i in x for j in y for k in z]

# print(f)

# Network building
net = tflearn.input_data(shape=[None, 3, 1])
net = tflearn.fully_connected(net, 200, activation='relu') # sigmoid
net = tflearn.fully_connected(net, 200, activation='relu')
net = tflearn.fully_connected(net, 2)
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.05)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(X, f, n_epoch=5)

predX, predY = [], []
for i in x:
    for j in y:
        for k in z:
            pred = model.predict(np.array([i,j,k]).reshape(-1,3,1))[0]
            predX.append(pred[0])
            predY.append(pred[1])

ab = [a[0] for a in f]
od = [a[1] for a in f]

plt.subplots()
plt.plot(range(len(ab)), ab, 'k', range(len(ab)), predX, 'b')
plt.subplots()
plt.plot(range(len(ab)), od, 'k', range(len(ab)), predY, 'b')
plt.show()