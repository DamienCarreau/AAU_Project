from scipy import interpolate
import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random, math
import model as md

def dnn(input_size, output_size):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    for _ in range(1): #1
        net = tflearn.fully_connected(net, 500, activation='relu')#500
    net = tflearn.fully_connected(net, output_size, activation='softsign')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.0001)#0.0001

    # Define model
    model = tflearn.DNN(net)

    return model

def train(X, y, model):
    # Start training (apply gradient descent algorithm)
    model.fit(X, y, n_epoch=40, snapshot_step=100)

    return model

def plot(X, y, model, display=True):
    predX, predY = [], []
    for i in X:
        pred = model.predict([i])[0]
        predX.append(pred[0])
        predY.append(pred[1]*math.pi/4) # Invert normalization

    ab = [a[0] for a in y]
    od = [a[1] for a in y]

    plt.subplots()
    plt.plot(range(len(ab)), ab, 'k', range(len(ab)), predX, 'b')
    plt.subplots()
    plt.plot(range(len(ab)), od, 'k', range(len(ab)), predY, 'b')
    if display:
        plt.show()

def predict(model, filename, plotMPC=False, display=True, validate=True):
    x, y = [], []
    set = np.load(f'data/{filename}.npy', allow_pickle=True)
    input = set[:,0]
    output = set[:,1]
    data = np.load(f'data/{filename}_data.npy', allow_pickle=True)
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
    expA, expD, obsA, obsD = [], [], [], []
    index = 0
    while x[-1] < cx[-1] and index < len(cx)-1:
        xref, ind, _ = md.calc_ref_trajectory(state, cx, cy, cyaw, _, sp, dl, ind)
        npState = []
        for i in xref[:4]:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,state.toArray()))
        npState = npState.reshape(-1, len(npState), 1)
        command = model.predict(npState)[0]
        state = md.update_state(state, command[0], command[1]*math.pi/4, DT) # Invert normalization
        x.append(state.x)
        y.append(state.y)
        if index < len(output):
            expA.append(output[index][0])
            expD.append(output[index][1])
        obsA.append(command[0])
        obsD.append(command[1]*math.pi/4)
        index += 1

    if plotMPC:
        mpcX, mpcY = [], []
        mpcState = md.State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
        for step in output:
            mpcState = md.update_state(mpcState, step[0], step[1]*math.pi/4, DT)
            mpcX.append(mpcState.x)
            mpcY.append(mpcState.y)

    plt.subplots()
    plt.plot(cx, cy, 'k', x, y, 'b')
    if plotMPC:
        plt.plot(mpcX, mpcY, 'g')
    plt.title(f'{filename} trajectory')
    plt.subplots()
    plt.plot(range(len(expA)), expA, 'k', range(len(obsA)), obsA, 'b')
    plt.title(f'{filename} acceleration')
    plt.subplots()
    plt.plot(range(len(expD)), expD, 'k', range(len(obsD)), obsD, 'b')
    plt.title(f'{filename} deltarate')
    if display:
        plt.show()

    if validate:
        print(validity(cx, cy, x, y))

def validity(cx, cy, x, y):
    tck = interpolate.splrep(cx, cy, s=0)
    cynew = interpolate.splev(x, tck, der=0)

    area = 0
    for index in range(len(x)-1):
        area += abs((x[index+1]-x[index])*(cynew[index]-y[index]))

    return area

load = True
retrain = False

training_set = np.load('data/training1000.npy', allow_pickle=True)
X = np.array([np.concatenate((i[0][:][:24], i[0][:][36:])) for i in training_set]).reshape(-1, 29, 1)
y = [i[1] for i in training_set]

model = dnn(input_size=len(X[0]), output_size=len(y[0]))

# model.load('data/x_ref_model.model')
# model.save('data/20_epoch_5l_1000n_0.0001LR_model.model')

if load == False:
    model = train(X, y, model)
    model.save('data/x_ref_model.model')
else:
    model.load('data/x_ref_model.model')
    if retrain:
        model = train(X, y, model)
        model.save('data/x_ref_model.model')

predict(model, 'sample2', plotMPC=False, display=False)
predict(model, 'trainingRef', plotMPC=False, display=False)
plt.show()
