import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


# generate cosine function
x = np.linspace(-np.pi,np.pi,10000).reshape(-1, 1)
y = np.sin(x)


# Network building
net = tflearn.input_data(shape=[None,1])
net = tflearn.fully_connected(net, 100, activation='softsign') # sigmoid
# net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 100, activation='softsign')
# net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 1)
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.01)


# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(x, y, n_epoch=10)

sum = 0
for _ in range(20):
    p = [[random.random()*np.pi*2-np.pi]]
    # print(p)
    q = model.predict(p)
    # print(q)
    sum += abs(np.sin(p[0][0]) - q[0][0])

print(sum/20)

pred = []
for i in x:
    pred.append(model.predict([i])[0][0])

plt.plot(x,y, 'k', x, pred, 'b')
plt.show()