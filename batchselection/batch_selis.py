#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""





import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as pyplot
import pylab as p
import math
import lasagne
import random
from bisect import bisect_right
import glob
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import matplotlib.pyplot as plt

import batch_norm
#from pylearn2.datasets.zca_dataset import ZCA_Dataset
#from pylearn2.utils import serial

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))  # 50k
    X_val = X_val.reshape((-1, 1, 28, 28))      # 10k
    X_test = X_test.reshape((-1, 1, 28, 28))    # 10k

    if (0):
        ntrain = 3000
        nvalid = 3000
        ntest = 3000
        X_train = X_train[1:ntrain]
        y_train = y_train[1:ntrain]
        X_val = X_val[1:nvalid]
        y_val = y_val[1:nvalid]
        X_test = X_test[1:ntest]
        y_test = y_test[1:ntest]

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None, nettype = 1):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    alpha = .15
    epsilon = 1e-4

    nfilters = 6
    usefullnet = 0
    fullnetsize = 0
    fulldrop = 0.5
    if (nettype == 1):  # small net
        nfilters = 6
        usefullnet = 0
        fullnetsize = 0
        fulldrop = 0

    if (nettype == 2):  # bigger net
        nfilters = 32
        usefullnet = 1
        fullnetsize = 256
        fulldrop = 0.5


    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=nfilters, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    if (1):
        network = batch_norm.BatchNormLayer(
                network,
                epsilon=epsilon,
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=nfilters, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.identity)
    if (1):
        network = batch_norm.BatchNormLayer(
                network,
                epsilon=epsilon,
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))



    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    if (usefullnet == 1):
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=fullnetsize,
                nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.0),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def test(model='cnn', num_epochs=50, bs_begin=16, bs_end=16, fac_begin=100, fac_end=100, pp1 = 0, pp2 = 0, alg=1, nettype=1, adapt_type = 0, irun = 1):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var,nettype)
    else:
        print("Unrecognized model type %r." % model)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    npww = np.empty(64)
    npww.fill(1.0)
    ww = theano.shared(npww)
    loss = loss * ww
    loss = loss.mean()

    losses = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    losses_fn = theano.function([input_var, target_var], losses)
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    if (alg == 1):  # adadelta with default parameters as given in lasagne
        updates = lasagne.updates.adadelta(loss, params, learning_rate=1.0, rho=0.95, epsilon=1e-06)
        algname = 'adadelta'
    if (alg == 2):  # adam with default parameters as given in lasagne
        updates = lasagne.updates.adam(loss, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        algname = 'adam'

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_fn_losses = theano.function([input_var, target_var], losses, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")

    bfs = []                    # to store last known loss for each datapoint
    ntraining = len(X_train)    # number of training datapoints
    # bfs = [1e+10]*ntraining     # set last known loss to a big number
    bfs = np.ndarray((ntraining,2))
    l = 0
    for l in range(0,ntraining):
        bfs[l][0] = 1e+10
        bfs[l][1] = l

    prob = [None]*ntraining     # to store probabilies of selection, prob[i] for i-th ranked datapoint
    sumprob = [None]*ntraining  # to store sum of probabilies from 0 to i-th ranked point

    # sort_index = np.argsort(bfs)[::-1]  # ind

    filename = algname + "_{}_{}_{}_{}_{}_{}_{}".format(irun, nettype, bs_end, pp1, pp2, fac_begin, fac_end) + ".txt"

    mult_bs = math.exp(math.log(bs_end/bs_begin)/num_epochs)
    mult_fac = math.exp(math.log(fac_end/fac_begin)/num_epochs)

    sorting_evaluations_period = 100   # increase it if sorting is too expensive
    sorting_evaluations_ago = 2*sorting_evaluations_period



    start_time0 = time.time()
    wasted_time = 0         # time wasted on computing training and validation losses/errors
    best_validation_error = 1e+10
    best_predicted_test_error = 1e+10
    myfile=open(filename, 'w+', 0)
    lastii = 0
    curii = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()
        fac = fac_begin * math.pow(mult_fac, epoch)
        if (adapt_type == 0):   # linear
            bs = bs_begin + (bs_end - bs_begin)*(float(epoch)/float(num_epochs-1))
        if (adapt_type == 1):   # exponential
            bs = bs_begin * math.pow(mult_bs, epoch)
        bs = int(math.floor(bs))

        if (fac == 1):
            for batch in iterate_minibatches(X_train, y_train, bs, shuffle=True):
                inputs, targets = batch
                batch_err = train_fn(inputs, targets)
        else:
            mult = math.exp(math.log(fac)/ntraining)
            for i in range(0, ntraining):
                if (i == 0):    prob[i] = 1.0
                else:           prob[i] = prob[i-1]/mult
            psum = sum(prob)
            prob =[v/psum for v in prob]
            for i in range(0, ntraining):
                if (i == 0):    sumprob[i] = prob[i]
                else:           sumprob[i] = sumprob[i-1] + prob[i]

            stop = 0
            iter = 0

            while (stop == 0):
                indexes = []
                wrt_sorted = 0
                if (epoch > 0):
                    wrt_sorted = 1
                    if (sorting_evaluations_ago >= sorting_evaluations_period):
                        bfs = bfs[bfs[:,0].argsort()[::-1]]
                        sorting_evaluations_ago = 0

                stop1 = 0
                while (stop1 == 0):
                    index = iter
                    if (wrt_sorted == 1):
                        randpos = min(random.random(), sumprob[-1])
                        index = bisect_right(sumprob, randpos)  # O(log(ntraining)), cheap
                    indexes.append(index)
                    iter = iter + 1
                    if (len(indexes) == bs) or (iter == len(X_train)):
                        stop1 = 1
                sorting_evaluations_ago = sorting_evaluations_ago + bs
                if (iter == len(X_train)):
                    stop = 1

                idxs = []
                for idx in indexes:
                    idxs.append(bfs[idx][1])

                wwnew = np.empty(len(indexes))
                if (1):
                    sumselprob = 0
                    for idx in indexes:
                        sumselprob = sumselprob + prob[idx]

                    i = 0
                    meanselprob = sumselprob/len(indexes)
                    for idx in indexes:
                        wwnew[i] = 1.0/(prob[idx]/ meanselprob)
                        i = i + 1
                else:
                    wwnew.fill(1.0)
                ww.set_value(wwnew)

                batch = X_train[idxs], y_train[idxs]
                inputs, targets = batch
                losses = train_fn_losses(inputs, targets)
                meanloss = np.mean(losses)
                i = 0
                for idx in indexes:
                    bfs[idx][0] = losses[i]
                    i = i + 1

                #if (1):
                curii = curii + len(idxs)

                if (pp1 > 0):
                    if (curii - lastii > ntraining/pp1):
                        lastii = curii
                        stopp = 0
                        iii = 0
                        bs_here = 500
                        maxpt = int(len(X_train)*pp2)
                        while (stopp == 0):
                            indexes = []
                            stop1 = 0
                            while (stop1 == 0):
                                index = iii
                                indexes.append(index)
                                iii = iii + 1
                                if (len(indexes) == bs_here) or (iii == maxpt):
                                    stop1 = 1

                            if (iii == maxpt):
                                stopp = 1

                            idxs = []
                            for idx in indexes:
                                idxs.append(bfs[idx][1])
                            batch = X_train[idxs], y_train[idxs]
                            inputs, targets = batch
                            losses = losses_fn(inputs, targets)
                            i = 0
                            for idx in indexes:
                                bfs[idx][0] = losses[i]
                                i = i + 1


        if (1): # otherwise report time only
            start_time_wasted0 = time.time()
            # a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_batches = 0
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            # a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            cur_valid_error = 100 - val_acc / val_batches * 100
            if (cur_valid_error < best_validation_error):
                best_validation_error = cur_valid_error
                test_err = 0
                test_acc = 0
                test_batches = 0
                for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    test_err += err
                    test_acc += acc
                    test_batches += 1
                best_predicted_test_error = 100 - test_acc / test_batches * 100

            start_time_wasted1 = time.time()
            epoch_wasted_time = start_time_wasted1 - start_time_wasted0
            wasted_time = wasted_time + epoch_wasted_time
            # Then we print the results for this epoch:
            print("Epoch {} of {}".format(epoch + 1, num_epochs))
            curtime = time.time()
            epoch_learning_time = curtime - start_time - epoch_wasted_time
            epoch_total_time = curtime - start_time
            total_learning_time = curtime - start_time0 - wasted_time
            total_time = curtime - start_time0
            print("epoch learning time {:.3f}s, epoch total time {:.3f}s, total learning time {:.3f}s, total time {:.3f}s".format(epoch_learning_time, epoch_total_time, total_learning_time, total_time))
            print("{}_{:.6f}".format(bs,fac))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  train error:\t\t{:.3f} %".format(100 - train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation error:\t\t{:.3f} % , test error:\t\t{:.3f} % ".format(cur_valid_error,best_predicted_test_error))
            myfile.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(epoch + 1, train_err / train_batches,
                                        val_err / val_batches, cur_valid_error, best_predicted_test_error, total_learning_time, total_time))

        else:
             print("Epoch {} of {} took {:.3f}s, total {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time, time.time() - start_time0))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        100 - test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))
def main():

    nettype = 2     # 1 - small network with 6 filters, 2 - bigger network with 32 filters, see function build_cnn for more details
    num_epochs = 50  # maximum number of epochs
    alg = 1         # 1 - AdaDelta, 2 - Adam, see function test for more details

    #bs_begin = 64   # batch size at epoch 0
    #bs_end = 64     # batch size at epoch 'num_epochs'
    #fac_begin = 1   # selection pressure at at epoch 0
    #fac_end = 1     # selection pressure at at 'num_epochs'
    adapt_type = 1  # 0 - linear, 1 - exponential change of batch size from bs_begin to bs_end as a function of epoch index

    if (1):
        nettype = 2
        run_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        alg_vals = [1, 2]
        pp_scenarios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        bs_vals = [64]
        for irun in run_vals:
            for bs in bs_vals:
                bs_begin = bs
                bs_end = bs
                for scenario in pp_scenarios:
                    if (scenario == 1):  fac_begin = 1;      fac_end = 1;       pp1 = 0;    pp2 = 0;
                    if (scenario == 2):  fac_begin = 1.01;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;

                    if (scenario == 3):  fac_begin = 1e+2;   fac_end = 1e+2;    pp1 = 0;    pp2 = 0;
                    if (scenario == 4):  fac_begin = 1e+2;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;
                    if (scenario == 5):  fac_begin = 1e+8;   fac_end = 1e+8;    pp1 = 0;    pp2 = 0;
                    if (scenario == 6):  fac_begin = 1e+8;   fac_end = 1.01;    pp1 = 0;    pp2 = 0;

                    if (scenario == 7):  fac_begin = 1e+2;   fac_end = 1e+2;    pp1 = 0.5;    pp2 = 1.0;
                    if (scenario == 8):  fac_begin = 1e+2;   fac_end = 1.01;    pp1 = 0.5;    pp2 = 1.0;
                    if (scenario == 9):  fac_begin = 1e+8;   fac_end = 1e+8;    pp1 = 0.5;    pp2 = 1.0;
                    if (scenario == 10):  fac_begin = 1e+8;   fac_end = 1.01;    pp1 = 0.5;    pp2 = 1.0;


                    for alg in alg_vals:
                        test('cnn', num_epochs, bs_begin, bs_end, fac_begin, fac_end, pp1, pp2, alg, nettype, adapt_type, irun)

    

if __name__ == '__main__':
    main()
