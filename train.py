from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.shared_randomstreams import RandomStreams

from ConvRelu import ConvReluPoolLayer
from Ridge import RidgeRegressionLayer
from utils import load_data

batch_size = 600
learning_rate=0.0003
n_epochs=5000

datasets = load_data()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

index = T.lscalar()

x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
y = T.matrix('y', dtype=theano.config.floatX)  # the labels are presented as 1D vector of
                    # [int] labels
######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')

layer0_input = x.reshape((batch_size, 6, 10, 7))

layer0 = ConvReluPoolLayer(
    input=layer0_input,
    filter_shape=(24, 6, 2, 2),
    volume_shape=(batch_size, 6, 10, 7)
)
layer1 = ConvReluPoolLayer(
    input=layer0.output,
    filter_shape=(48, 24, 3, 2),
    volume_shape=(batch_size, 24, 9, 6)
)
layer2 = ConvReluPoolLayer(
    input=layer1.output,
    filter_shape=(96, 48, 2, 2),
    volume_shape=(batch_size, 48, 7, 5),
    mp2=True
)
layer3 = ConvReluPoolLayer(
    input=layer2.output,
    filter_shape=(192, 96, 3, 2),
    volume_shape=(batch_size, 96, 3, 2)
)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
layer4_input = layer3.output.flatten(2)
layer4 = RidgeRegressionLayer(input=layer4_input, n_in=192, n_out=32, penalty=0.1)

test = False

if test:
    test_set_x, test_set_y = load_data()[1]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.get_value()
    n = test_set_x.shape[0]
    f = open('2700.pkl', 'rb')
    layer4 = pickle.load(f)
    layer3 = pickle.load(f)
    layer2 = pickle.load(f)
    layer1 = pickle.load(f)
    layer0 = pickle.load(f)
    f.close()
    px = T.tensor4('px', dtype=theano.config.floatX)
    player0_input = px.reshape((n, 6, 10, 7))
    player0 = ConvReluPoolLayer(
        input=player0_input,
        filter_shape=(48, 6, 2, 2),
        volume_shape=(n, 6, 10, 7),
        W=layer0.W,
        b=layer0.b
    )
    player1 = ConvReluPoolLayer(
        input=player0.output,
        filter_shape=(96, 48, 3, 2),
        volume_shape=(n, 48, 9, 6),
        W=layer1.W,
        b=layer1.b
    )
    player2 = ConvReluPoolLayer(
        input=player1.output,
        filter_shape=(192, 96, 2, 2),
        volume_shape=(n, 96, 7, 5),
        mp2=True,
        W=layer2.W,
        b=layer2.b
    )
    player3 = ConvReluPoolLayer(
        input=player2.output,
        filter_shape=(384, 192, 3, 2),
        volume_shape=(n, 192, 3, 2),
        W=layer3.W,
        b=layer3.b
    )
    player4_input = player3.output.flatten(2)
    player4 = RidgeRegressionLayer(input=player4_input, n_in=384, n_out=32, penalty=0.1, W=layer4.W, b=layer4.b)

    predict_model = theano.function([],outputs=player4.y_pred, givens={
            px: test_set_x
        }, on_unused_input='warn')
    print("start predicting...")
    prediction = predict_model()
    print((((test_set_y - prediction) ** 2) / n).sum())
    # import csv
    # with open('results.csv', 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow(['Id','Prediction'])
    #     Id = 1
    #     for row in prediction:
    #         for col in row:
    #             writer.writerow([Id] + [col])
    #             Id += 1
    exit()

# the cost we minimize during training is the NLL of the model
cost = layer4.cost(y)

validate_model = theano.function(
    [index],
    layer4.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)
rs = RandomStreams()
# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.

# * T.cast(rs.binomial(size=T.shape(param_i), p=0.75), theano.config.floatX))
updates = [(params[0], params[0] - learning_rate * grads[0]), (params[1], params[1] - learning_rate * grads[1])] + [
    (params[i], (params[i] - learning_rate * grads[i]) * T.cast(rs.binomial(size=T.shape(params[i]), p=(0.5 + 0.1 * (i // 2))), theano.config.floatX))
    for i in range(2, len(params))
]

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

print('... training the model')
# early-stopping parameters
improvement_threshold = 0.995  # a relative improvement of this much is
                              # considered significant
validation_frequency = n_train_batches
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = np.inf
start_time = timeit.default_timer()

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):

        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation MSE %f ' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                best_validation_loss = this_validation_loss

                print(
                    (
                        '     epoch %i, minibatch %i/%i, validation MSE of'
                        ' best model %f'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        best_validation_loss
                    )
                )
                f = open('best_model.pkl', 'wb')
                for layer in [layer4, layer3, layer2, layer1, layer0]:
                    pickle.dump(layer, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

end_time = timeit.default_timer()
print(
    (
        'Optimization complete with best validation score of %f %%,'
    )
    % (best_validation_loss * 100.)
)
print('The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time)))
print(('The code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)