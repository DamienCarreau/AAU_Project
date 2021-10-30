import tflearn
from matplotlib import pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import math
import numpy as np
import model as md

# Learning rate
LR = 1e-3

DT = 0.1


def neural_network_model(input_size):
    # The input is a vector of length input_size
    network = input_data(shape=[None, input_size, 1], name='input')

    # First layer, 128 nodes
    # activation function : ReLU (Rectified Linear Unit) activation function
    # f(x) = max(0,x)
    network = fully_connected(network, 128, activation='relu')
    # We have a probability of drop of 0.8
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # output layer of size 2
    network = fully_connected(network, 2, activation='softmax')
    # we compare our solution with the expected solution
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    # We assemble ou model
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0][:] for i in training_data]).reshape(-1, len(training_data[0][0][:]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


# trainning_set = np.load('data/trainning1000.npy', allow_pickle=True)
# model = train_model(trainning_set)
# model.save('data/model1000.model')

model = neural_network_model(input_size=41)
model.load('data/model1000.model')


def predict():
    x, y, cx, cy, ea, ed, oa, od = [], [], [], [], [], [], [], []
    set = np.load('data/sample1.npy', allow_pickle=True)
    input = set[:, 0]
    output = set[:,1]
    cx = set[:,2]
    cy = set[:,3]
    cyaw = set[:,4]
    ck = set[:,5]
    sp = set[:,6]
    dl = 1
    ind = 0

    x0 = input[0][36:41]
    state = md.State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
    x.append(state.x)
    y.append(state.y) 
    for index in range(len(set)):
        xref, ind, _ = md.calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, ind)
        npState = []
        for i in xref:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,state.toArray()))
        npState = npState.reshape(-1, len(npState), 1)
        command = model.predict(npState)[0]
        state = md.update_state(state, command[0], command[1], DT)
        x.append(state.x)
        y.append(state.y)
        ea.append(output[index][0])
        ed.append(output[index][1])
        oa.append(command[0])
        od.append(command[1])
        # print(f'{command[0]} <=> {output[index][0]} || {command[1]} <=> {output[index][1]}')

    # plt.plot(range(len(ea)), np.array(ea)-np.array(oa))
    # plt.show()

    plt.plot(x, y, 'b', cx, cy, 'g')
    plt.show()


predict()
