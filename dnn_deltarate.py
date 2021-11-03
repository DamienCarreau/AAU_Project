import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import model as md

def dnn(input_size, output_size):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    for _ in range(50): #5
        net = tflearn.fully_connected(net, 500, activation='relu')#500
    net = tflearn.fully_connected(net, output_size)
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.0001)#0.0001

    # Define model
    model = tflearn.DNN(net)

    return model

def train(X, y, model):
    # Start training (apply gradient descent algorithm)
    model.fit(X, y, n_epoch=10, snapshot_step=100)

    return model

def plot(X, y, model):
    horizon = 2000
    predX = []
    for i in X[:horizon]:
        pred = model.predict([i])[0]
        predX.append(pred[0])

    ab = [a[0] for a in y[:horizon]]

    plt.subplots()
    plt.plot(range(len(ab)), ab, 'k', range(len(ab)), predX, 'b')
    plt.show()

load = True

training_set = np.load('data/training1000.npy', allow_pickle=True)
X = np.array([i[0][:] for i in training_set]).reshape(-1, len(training_set[0][0][:]), 1)
y = [[i[1][1]] for i in training_set]

model = dnn(input_size=len(X[0]), output_size=len(y[0]))

if load == False:
    model = train(X, y, model)
    model.save('data/del_model.model')
else:
    model.load('data/del_model.model')

plot(X, y, model)