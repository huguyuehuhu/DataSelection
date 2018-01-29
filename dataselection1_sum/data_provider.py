# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-20 下午8:13
"""
import sys
import os


class DataSet:
    def __init__(self, num_examples, n_class, shape, x, y):
        self.num_examples = num_examples
        self.n_class = n_class
        self.data = x
        self.label = y
        self.shape = shape


class Data:
    def __init__(self, train_data, val_data, test_data):
        self.train = train_data
        self.validation = val_data
        self.test = test_data


def load_dataset(debug=False, data_name='MNIST'):
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

    if data_name == 'MNIST':
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
        x_train, y_train = data[0]
        x_val, y_val = data[1]
        x_test, y_test = data[2]

        # The inputs come as vectors, we reshape them to monochrome 2D images,
        # according to the shape convention: (examples, channels, rows, columns)
        # x_train = x_train.reshape((-1, 1, 28, 28))  # 50k
        # x_val = x_val.reshape((-1, 1, 28, 28))      # 10k
        # x_test = x_test.reshape((-1, 1, 28, 28))    # 10k

        if debug:
            ntrain = 3000
            nvalid = 3000
            ntest = 3000
            x_train = x_train[1:ntrain]
            y_train = y_train[1:ntrain]
            x_val = x_val[1:nvalid]
            y_val = y_val[1:nvalid]
            x_test = x_test[1:ntest]
            y_test = y_test[1:ntest]

    # The targets are int64, we cast them to int8 for GPU compatibility.
    import numpy as np
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    num_examples = x_train.shape[0]
    n_class = len(np.unique(y_train))
    temp = x_train.shape
    shape = temp[1:]
    train_data = DataSet(num_examples, n_class, shape, x_train, y_train)

    num_examples = x_val.shape[0]
    n_class = len(np.unique(y_val))
    temp = x_val.shape
    shape = temp[1:]
    val_data = DataSet(num_examples, n_class, shape, x_val, y_val)

    num_examples = x_test.shape[0]
    n_class = len(np.unique(y_test))
    temp = x_test.shape
    shape = temp[1:]
    test_data = DataSet(num_examples, n_class, shape, x_test, y_test)

    # We just return all the arrays in order
    # (It doesn't matter how we do this as long as we can read them again.)
    return Data(train_data, val_data, test_data)


def func():
    pass


if __name__ == '__main__':
    load_dataset(debug=False, data_name='MNIST')
