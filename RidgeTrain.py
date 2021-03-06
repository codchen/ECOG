from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit
import math

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
n_epochs=1000

datasets = load_data(reshape=False)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

index = T.lscalar()

x = T.matrix('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
y = T.matrix('y', dtype=theano.config.floatX)  # the labels are presented as 1D vector of


layer0 = RidgeRegressionLayer(input=x, n_in=420, n_out=32, penalty=1)

test = True

if test:
    test_set_x = load_data(test=True,reshape=False).get_value()
    n = test_set_x.shape[0]
    f = open('best_model.pkl', 'rb')
    layer0 = pickle.load(f)
    f.close()
    px = T.matrix('px', dtype=theano.config.floatX)
    player0_input = px.reshape((n, 420))
    player0 = RidgeRegressionLayer(input=player0_input, n_in=420, n_out=32, penalty=1, W=layer0.W, b=layer0.b)

    predict_model = theano.function([],outputs=player0.y_pred, givens={
            px: test_set_x
        }, on_unused_input='warn')
    print("start predicting...")
    prediction = predict_model()
    import csv
    with open('results_ridge.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id','Prediction'])
        Id = 1
        for row in prediction:
            for col in row:
                writer.writerow([Id] + [col])
                Id += 1
    exit()

cost = layer0.cost(y)

validate_model = theano.function(
    [index],
    layer0.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)
rs = RandomStreams()
# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.

# * T.cast(rs.binomial(size=T.shape(param_i), p=0.75), theano.config.floatX))
updates = [
    (params[i], (params[i] - learning_rate * grads[i])) # * T.cast(rs.binomial(size=T.shape(params[i]), p=(0.5 + 0.1 * (i // 2))), theano.config.floatX))
    for i in range(len(params))
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
prev_val_loss = 0
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
                for layer in [layer0]:
                    pickle.dump(layer, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

            if abs(prev_val_loss - this_validation_loss) < 0.5 and learning_rate < 100:
            	learning_rate *= 10
            else:
            	learning_rate = 0.0003
            prev_val_loss = this_validation_loss

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