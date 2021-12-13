import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import tflearn
import tensorflow as tf
import model as md
import DNN
import random, math, time
import cubic_spline_planner
import curve
import generator

def getNewCurve():
    (xa, ya) = curve.createCurveRec(DNN.TRACK_LENGTH)
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(xa,ya)

    sp = generator.calc_speed_profile(cx, cy, cyaw, generator.TARGET_SPEED)

    return cx, cy, cyaw, sp



if __name__ == '__main__':
    dnn = ['dnn_1_5_40e_96', 'dnn_1_50_40e_100', 'dnn_1_250_40e_100', 'dnn_1_450_40e_100', 'dnn_1_500_70e_86', 'dnn_2_150_40e_90', 'dnn_2_150_40e_92', 'dnn_2_250_40e_98', 'dnn_2_450_40e_92']
    lstm = ['lstm_1_150_10e_100', 'lstm_1_250_10e_96', 'lstm_1_250_10e_98', 'lstm_1_350_10e_100', 'lstm_1_450_5e_94']
    gru = ['gru_1_50_5e_100', 'gru_1_150_5e_98', 'gru_1_250_5e_100']

    results = []
    
    for _ in range(1000):
        results = []
        cx, cy, cyaw, sp = getNewCurve()
        initial_state = md.State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

        # MPC
        startTime = time.time()
        t, x, y, yaw, v, d, a = generator.do_simulation(cx, cy, cyaw, [], sp, 1.0, initial_state)

        results.append('MPC')
        results.append(str(DNN.validate(cx, cy, x, y)))
        results.append(str(time.time()-startTime))

        cyaw = md.smooth_yaw(cyaw)

        for nn in dnn:
            tf.compat.v1.reset_default_graph()
            model = DNN.dnn(input_size=29, output_size=2, layers=int(nn.split('_')[1]), nodes=int(nn.split('_')[2]))
            model.load(f'data/models/{nn}')
            startTime = time.time()
            initial_state = md.State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
            x, y = DNN.predictWithoutValidate(model, cx, cy, cyaw, sp, initial_state, md.DT)
            results.append(nn)
            results.append(str(DNN.validate(cx, cy, x, y)))
            results.append(str(time.time()-startTime))

        for nn in lstm:
            tf.compat.v1.reset_default_graph()
            model = DNN.lstm(input_size=29, output_size=2, nodes=int(nn.split('_')[2]))
            model.load(f'data/models/{nn}')
            startTime = time.time()
            initial_state = md.State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
            x, y = DNN.predictWithoutValidate(model, cx, cy, cyaw, sp, initial_state, md.DT)
            results.append(nn)
            results.append(str(DNN.validate(cx, cy, x, y)))
            results.append(str(time.time()-startTime))

        for nn in gru:
            tf.compat.v1.reset_default_graph()
            model = DNN.gru(input_size=29, output_size=2, nodes=int(nn.split('_')[2]))
            model.load(f'data/models/{nn}')
            startTime = time.time()
            initial_state = md.State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
            x, y = DNN.predictWithoutValidate(model, cx, cy, cyaw, sp, initial_state, md.DT)
            results.append(nn)
            results.append(str(DNN.validate(cx, cy, x, y)))
            results.append(str(time.time()-startTime)) 

        f = open('data/validation/efficiency.txt', 'a')
        f.write(",".join(results)+'\n')
        f.close()