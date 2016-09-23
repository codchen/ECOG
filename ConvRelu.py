from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal import pool

from math import sqrt

class ConvReluPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, filter_shape, volume_shape, mp2=False, W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert filter_shape[1] == volume_shape[1]
        self.input = input

        n = volume_shape[0]
        if W is None:
            self.W = theano.shared(
                value=(np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * sqrt(2.0/n)).astype(theano.config.floatX),
                borrow=True
            )
        else:
            self.W = W

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=volume_shape
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if not mp2:
            self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            pooled_out = pool.pool_2d(
                input=conv_out,
                ds=(2,2),
                ignore_border=True
            )
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input