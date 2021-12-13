from numpy.core.numeric import NaN
from scipy import interpolate
import tflearn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random, math, time
import model as md

TRACK_LENGTH = 200
MAX_DEVIATION = 0.02*TRACK_LENGTH
EXPECTED_PREDICTION_END = TRACK_LENGTH - 5

def dnn(input_size, output_size, layers=1, nodes=50):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    for _ in range(layers):
        net = tflearn.fully_connected(net, nodes, activation='relu')
    net = tflearn.fully_connected(net, output_size, activation='softsign')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001)

    # Define model
    model = tflearn.DNN(net)

    return model

def lstm(input_size, output_size, nodes=50):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    net = tflearn.lstm(net, nodes)
    net = tflearn.fully_connected(net, output_size, activation='tanh')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001)

    # Define model
    model = tflearn.DNN(net)

    return model

def gru(input_size, output_size, nodes=50):
    # Network building
    net = tflearn.input_data(shape=[None, input_size, 1])
    net = tflearn.gru(net, nodes)
    net = tflearn.fully_connected(net, output_size, activation='tanh')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001)

    # Define model
    model = tflearn.DNN(net)

    return model

def train(X, y, model, n_epoch=10):
    # Start training (apply gradient descent algorithm)
    model.fit(X, y, n_epoch=n_epoch)

    return model

def predict(model, cx, cy, cyaw, sp, x0, DT):
    x, y = [],[]
    dl = 1.0
    state = md.State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
    ind, _ = md.calc_nearest_index(state, cx, cy, cyaw, 0)
    cyaw = md.smooth_yaw(cyaw)

    f = interpolate.interp1d(cx, cy)
    x.append(state.x)
    y.append(state.y)
    index = 0
    while x[-1] < cx[-md.T-1] and index < 1.5*len(cx):
        xref, ind, _ = md.calc_ref_trajectory(state, cx, cy, cyaw, _, sp, dl, ind)
        npState = []
        for i in xref[:4]:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,state.toArray()))
        npState = npState.reshape(-1, len(npState), 1)
        command = model.predict(npState)[0]
        state = md.update_state(state, command[0], command[1]*math.pi/4, DT) # Invert normalization
        if state.x >= cx[-1]: # Interpolation will crash if x is above the cx range
            return x,y,True
        # Second condition : Does the prediction turn around ?
        # Thrid condition : Is the deviation always less than MAX_DEVIATION
        if state.x < x[-1] or abs(state.y-f(state.x)) > MAX_DEVIATION:
            return x,y,False
        x.append(state.x)
        y.append(state.y)
        index += 1
    
    # First condition : Does the prediction end after EXPECTED_PREDICTION_END
    if state.x < EXPECTED_PREDICTION_END:
        return x,y,False
    return x,y,True

def predictWithoutValidate(model, cx, cy, cyaw, sp, initial_state, DT):
    x, y = [],[]
    dl = 1.0
    state = initial_state
    ind, _ = md.calc_nearest_index(state, cx, cy, cyaw, 0)
    cyaw = md.smooth_yaw(cyaw)
    
    x.append(state.x)
    y.append(state.y)
    index = 0
    while x[-1] < cx[-md.T-1] and index < 1.5*len(cx):
        xref, ind, _ = md.calc_ref_trajectory(state, cx, cy, cyaw, _, sp, dl, ind)
        npState = []
        for i in xref[:4]:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,state.toArray()))
        npState = npState.reshape(-1, len(npState), 1)
        command = model.predict(npState)[0]
        state = md.update_state(state, command[0], command[1]*math.pi/4, DT) # Invert normalization
        if state.x >= cx[-1]: # Interpolation will crash if x is above the cx range
            return x,y
        x.append(state.x)
        y.append(state.y)
        index += 1
    return x,y

def validate(cx, cy, x, y):
    # 1 : The prediction must end around 200m
    # 2 : Can't go backward : x must increase
    if x[-1] < EXPECTED_PREDICTION_END or not np.all(np.diff(x) >= 0):
        return False

    # 3 : Return False if the maximum of the difference between cy and y for a same x is bigger than 2% of the track_length
    x = np.array(x)
    x[x > cx[-1]] = cx[-1]
    try:
        f = interpolate.interp1d(cx, cy)
        if np.abs(y-f(x)).max() > MAX_DEVIATION:
            return False
    except:
        f = open('./data/validation/log.txt', 'a')
        f.write(f'{time.time()}- An error occured in validate function! \n')
        f.close()
        return False
    return True