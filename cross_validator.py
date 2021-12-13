import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import tflearn
import tensorflow as tf
import model as md
import DNN
import random, math, time

nbTraining = 5
correctPrediction = []
prediction = []

model = None

for layers in range(1,6):
    for nodes in range(50,550,100):
        startTime = time.time()
        correctPrediction = []
        prediction = []
        for validitySet in range(nbTraining):
            tf.compat.v1.reset_default_graph()
            model = DNN.dnn(input_size=29, output_size=2, layers=layers, nodes=nodes)

            # We train the model
            print('We begin a new training !')
            # model.load('data/x_ref_model.model')
            for index in range(nbTraining):
                if index == validitySet:
                    continue
                training_set = np.load(f'data/training/training{index}.npy', allow_pickle=True)
                X = np.array([np.concatenate((i[0][:][:24], i[0][:][36:])) for i in training_set]).reshape(-1, 29, 1)
                y = [i[1] for i in training_set]

                model = DNN.train(X, y, model, n_epoch=40)

            # We validate the model with the validitySet
            print('Training is done, passing to the validation !')
            f = open('./data/validation/log.txt', 'a')
            f.write(f'{time.time()} - validation for l={layers}, n={nodes}, set={validitySet} begin !\n')
            f.close()
            validity_set = np.load(f'data/training/training{validitySet}.npy', allow_pickle=True)
            X = np.array([np.concatenate((i[0][:][:24], i[0][:][36:])) for i in validity_set]).reshape(-1, 29, 1)
            y = [i[1] for i in validity_set]

            validity_data = np.load(f'data/training/training{validitySet}_data.npy', allow_pickle=True)

            pred = 0
            nbPred = 0
            for index in range(len(validity_data)):
                startIndex = 0
                if index > 0:
                    startIndex = validity_data[index-1][4]
                x0 = (X[startIndex][24:]).reshape(5)
                _,_,b = DNN.predict(model, validity_data[index][0], validity_data[index][1], validity_data[index][2], validity_data[index][3], x0, md.DT)
                if b:
                    pred += 1
                nbPred += 1

            if pred/nbPred > 0.8:
                model.save(f'./data/models/model_{pred/nbPred}_{layers}_{nodes}.nn')

            correctPrediction.append(pred)
            prediction.append(pred/nbPred)

        endTime = time.time()

        file = open('./data/validation/validation.txt', 'a')
        file.write(f'{layers},{nodes},{np.array(prediction).mean()},{correctPrediction},{prediction},{endTime-startTime}\n')
        file.close()