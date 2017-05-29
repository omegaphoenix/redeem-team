###############################################################################
# NN Blending
# Based on CS 155 sample code
###############################################################################

from math import sqrt
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

N_MODELS = 0
N_NEURONS = 70
N_PROBE = 1374739
N_23 = 3929436
BATCH_SIZE = N_PROBE
N_QUAL = 2749898
N_EPOCHS = None # Gets overwritten
EPOCH_LIST = range(1, 100, 10)
MEAN = 3.6095161972728063 # 3.6007
STD = 1.0813423177560841


def normal(val):
    return (val - MEAN) / STD

def unnormalize(val):
    tmp = STD * val + MEAN
    if tmp > 5:
        tmp = 5
    elif tmp < 1:
        tmp = 1
    return tmp

X_train, y_train, X_test, y_test = [], [], [], []

# all_data = []
# with open('data/um/all.dta') as data:
    # for i, line in enumerate(data):
        # val = float(line.split()[3])
        # if val != 0:
            # all_data.append(val)
    # all_data = np.asarray(all_data)
# MEAN = all_data.mean()
# STD = all_data.std()
# import pdb; pdb.set_trace()

with open('nn-blend/trainX.dta') as data:
    for i, line in enumerate(data):
        if i % N_23 == 0:
            X_train.append([])
        X_train[-1].append(normal(float(line)))
    X_train = np.asarray(X_train).transpose()

with open('data/um/23.dta') as data:
    for line in data:
        y_train.append(normal(float(line.split()[3])))
    y_train = np.asarray(y_train)

with open('nn-blend/testX.dta') as data:
    for i, line in enumerate(data):
        if i % N_PROBE == 0:
            N_MODELS += 1
            X_test.append([])
        X_test[-1].append(normal(float(line)))
    X_test = np.asarray(X_test).transpose()
print "{} models".format(N_MODELS)

with open('data/um/4.dta') as data:
    for line in data:
        y_test.append(normal(float(line.split()[3])))
    y_test = np.asarray(y_test)

X_out = []
with open('nn-blend/outX.dta') as data:
    for i, line in enumerate(data):
        if i % N_QUAL == 0:
            X_out.append([])
        X_out[-1].append(normal(float(line)))
    X_out = np.asarray(X_out).transpose()

def run():
    ## Create your own model here given the constraints in the problem
    model = Sequential()
    model.add(Dense(N_MODELS, input_dim=N_MODELS))
    model.add(Activation('linear'))
    # model.add(Dense(N_NEURONS))
    # model.add(Activation('tanh'))
    # model.add(Dense(N_NEURONS))
    # model.add(Activation('tanh'))
    model.add(Dense(N_NEURONS))
    model.add(Activation('tanh'))
    ## Once you one-hot encode the data labels, the line below should be predicting probabilities of each of the 10 classes
    ## e.g. it should read: model.add(Dense(10)), not model.add(Dense(1))
    model.add(Dense(1))
    # model.add(Activation('softmax'))

    ## Printing a summary of the layers and weights in your model
    model.summary()

    ## In the line below we have specified the loss function as 'mse' (Mean Squared Error) because in the above code we did not one-hot encode the labels.
    ## In your implementation, since you are one-hot encoding the labels, you should use 'categorical_crossentropy' as your loss.
    ## You will likely have the best results with RMS prop or Adam as your optimizer.  In the line below we use Adadelta
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    fit = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCHS,
        verbose=1)

    ## Printing the accuracy of our model, according to the loss function specified in model.compile above
    score = model.evaluate(X_test, y_test, verbose=0)
    rmse = sqrt(score[0])
    print('Test score (prmse):', rmse)

    predictions = model.predict(X_out)
    # import pdb; pdb.set_trace()

    print('Printing output for N_EPOCHS = {}'.format(N_EPOCHS))
    out_file = 'nn-blend/blend_{}epochs_{}neurons.out'.format(N_EPOCHS, N_NEURONS)
    with open(out_file, 'w') as f:
        for p in predictions:
            f.write('{}\n'.format(unnormalize(p[0])))

    predictions = model.predict(X_test)
    print('Printing 4.dta for N_EPOCHS = {}'.format(N_EPOCHS))
    test_file = 'nn-blend/blend_{}epochs_{}neurons-4.dta'.format(N_EPOCHS, N_NEURONS)
    with open(test_file, 'w') as f:
        for p in predictions:
            f.write('{}\n'.format(unnormalize(p[0])))

    return rmse, out_file

if __name__ == '__main__':
    sys.stderr.write('{} models\n'.format(N_MODELS))
    blends = []
    for i in EPOCH_LIST:
        N_EPOCHS = i
        blends.append(run())

    blends = sorted(blends, key=lambda model: model[0])
    for blend in blends:
        sys.stderr.write('{1}: {0}\n'.format(*blend))
    sys.stderr.write('\n')
