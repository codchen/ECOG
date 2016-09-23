from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T

class RidgeRegressionLayer(object):
    """
    Ridge Regression Class
    """

    def __init__(self, input, n_in, n_out, penalty=0.1, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W
        # initialize the biases b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        self.y_pred = T.dot(input, self.W) + self.b

        self.output = T.nnet.relu(self.y_pred)

        self.params = [self.W, self.b]

        self.input = input

        self.penalty = penalty

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        return (((self.y_pred - y) ** 2) / T.shape(self.input)[0]).sum()

    def cost(self, y):
        return self.errors(y) + ((self.W ** 2) / T.shape(self.input)[0]).sum()

    def cost_h(self):
        return ((self.W ** 2) / T.shape(self.input)[0]).sum()
