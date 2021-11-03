import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import model as md

def dnn(input_size, output_size):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    for _ in range(5): #5
        net = tflearn.fully_connected(net, 500, activation='relu')#500
    net = tflearn.fully_connected(net, output_size)
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.0001)#0.0001

    # Define model
    model = tflearn.DNN(net)

    return model

def train(X, y, model):
    # Start training (apply gradient descent algorithm)
    model.fit(X, y, n_epoch=1, snapshot_step=100)

    return model

def plot(X, y, model, display=True):
    horizon = 2000
    predX, predY = [], []
    for i in X[:horizon]:
        pred = model.predict([i])[0]
        predX.append(pred[0])
        predY.append(pred[1])

    ab = [a[0] for a in y[:horizon]]
    od = [a[1] for a in y[:horizon]]

    plt.subplots()
    plt.plot(range(len(ab)), ab, 'k', range(len(ab)), predX, 'b')
    plt.subplots()
    plt.plot(range(len(ab)), od, 'k', range(len(ab)), predY, 'b')
    if display:
        plt.show()

def predict(model, display=True):
    x, y = [], []
    set = np.load('data/sample2.npy', allow_pickle=True)
    input = set[:,0]
    output = set[:,1]
    data = np.load('data/sample2_data.npy', allow_pickle=True)
    cx = data[0]
    cy = data[1]
    cyaw = data[2]
    sp = data[3]
    dl = 1.0
    DT = 0.1
    x0 = input[0][36:41]
    state = md.State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
    ind, _ = md.calc_nearest_index(state, cx, cy, cyaw, 0)
    cyaw = md.smooth_yaw(cyaw)

    x.append(state.x)
    y.append(state.y) 
    for index in range(len(set)):
        xref, ind, _ = md.calc_ref_trajectory(state, cx, cy, cyaw, _, sp, dl, ind)
        npState = []
        for i in xref:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,state.toArray()))
        npState = npState.reshape(-1, len(npState), 1)
        command = model.predict(npState)[0]
        state = md.update_state(state, command[0], command[1], DT)
        x.append(state.x)
        y.append(state.y)
        # plt.cla()
        # plt.plot(cx[:ind], cy[:ind], 'k', x, y, 'b', state.x, state.y, 'xb', xref[0, :], xref[1, :], 'xr')
        # plt.pause(0.01)

    plt.subplots()
    plt.plot(cx, cy, 'k', x, y, 'b')
    if display:
        plt.show()

load = False

training_set = np.load('data/training1000.npy', allow_pickle=True)
X = np.array([i[0][:] for i in training_set]).reshape(-1, len(training_set[0][0][:]), 1)
y = [i[1] for i in training_set]

model = dnn(input_size=len(X[0]), output_size=len(y[0]))

if load == False:
    model = train(X, y, model)
    model.save('data/x_ref_model.model')
else:
    model.load('data/x_ref_model.model')

plot(X, y, model, display=False)
# predict(model, display=False)
plt.show()