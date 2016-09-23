from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T

from sklearn.cross_validation import train_test_split

def load_data(test=False, reshape=True):
    ''' 
    Loads the dataset
    '''
    #############
    # LOAD DATA #
    #############
    print('... loading data')
    if test:
        X_test = np.genfromtxt('test_X_ecog.csv', delimiter=',')
        X_test -= np.mean(X_test, axis=0)
        X_test /= np.std(X_test, axis=0)
        if reshape:
            X_test = np.reshape(X_test, (X_test.shape[0], 6, 10, 7))
        shared_x = theano.shared(np.asarray(X_test, dtype=theano.config.floatX), borrow=True)
        return shared_x
    X_train = np.genfromtxt('train_X_ecog.csv', delimiter=',')
    X_train -= np.mean(X_train, axis = 0)
    X_train /= np.std(X_train, axis = 0)
    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], 6, 10, 7))
    Y_train = np.genfromtxt('train_Y_ecog.csv', delimiter=',')
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    valid_set_x, valid_set_y = shared_dataset((X_val, Y_val))
    train_set_x, train_set_y = shared_dataset((X_train, Y_train))
    print(valid_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval