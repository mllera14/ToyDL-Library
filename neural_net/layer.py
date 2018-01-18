import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from tools.rng_tools import get_numpy_rng


class LinearLayer(object):
    """ Linear Regression Layer """
    @property
    def theta(self):
        return self.params[0]

    @property
    def bias(self):
        return self.params[1]

    def __init__(self, n_units, n_features):
        w = theano.shared(
            value=np.zeros((n_features, n_units), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        b = theano.shared(
            value=np.zeros((n_units,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.params = [w, b]

    def activation(self, inputs):
        return T.dot(inputs, self.theta) + self.bias


class SoftmaxLayer(object):
    """ Softmax Classification Layer """
    @property
    def theta(self):
        return self.params[0]

    @property
    def bias(self):
        return self.params[1]

    def __init__(self, n_units, n_features):
        w = theano.shared(
            value=np.zeros((n_features, n_units), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        b = theano.shared(
            value=np.zeros((n_units,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.params = [w, b]

    def activation(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.theta) + self.bias)


class DenseLayer(object):
    """ Fully Connected Layer """
    @property
    def theta(self):
        return self.params[0]

    @property
    def bias(self):
        return self.params[1]

    def __init__(self, n_units, n_features, rng=None, w=None, b=None, activation_function=T.tanh):
        rng = get_numpy_rng(rng)

        if w is None:
            epsilon = np.sqrt(6. / (n_units + n_features))
            w_values = np.asarray(
                rng.uniform(
                    low=-epsilon,
                    high=epsilon,
                    size=(n_features, n_units)
                ), dtype=theano.config.floatX
            )
            if activation_function == T.nnet.sigmoid:
                w_values *= 4.
            w = theano.shared(value=w_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_units,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [w, b]
        self._act_fn = activation_function

    def activation(self, inputs):
        return self._act_fn(T.dot(inputs, self.theta) + self.bias)


# noinspection PyPep8Naming
def LogisticLayer(n_units, n_features, rng=None, w=None, b=None):
    return DenseLayer(n_units, n_features, rng, w, b, T.tanh)


class ConvolutionPoolLayer(object):
    """ Convolution and Max Pooling Layer """
    def __init__(self, n_filters, n_input_fmaps, filter_shape, image_shape, pool_size=(2, 2), rng=None):

        self.filter_shape = (n_filters, n_input_fmaps, filter_shape[0], filter_shape[1])
        self.input_shape = (1, n_input_fmaps, image_shape[0], image_shape[1])
        self.pool_size = pool_size

        fan_in, fan_out = np.prod(filter_shape) * n_input_fmaps, n_filters * np.prod(filter_shape) / np.prod(pool_size)
        epsilon = np.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(
            np.asarray(rng.uniform(
                    low=-epsilon,
                    high=epsilon,
                    size=self.filter_shape
                ), dtype=theano.config.floatX
            ),
            borrow=True,
            name='W'
        )
        b = theano.shared(np.zeros((n_filters,), dtype=theano.config.floatX), 'b')

        self.W, self.b = W, b
        self.params = [self.W, self.b]

    @property
    def theta(self):
        return self.W

    @property
    def bias(self):
        return self.b

    # def set_batch_size(self, value):
    #     shape = self.image_shape
    #     self.image_shape = value, shape[1], shape[2], shape[3]

    def convolve(self, x):
        return conv2d(x, filters=self.W, input_shape=self.input_shape, filter_shape=self.filter_shape)

    def max_pool(self, x):
        return pool_2d(x, ws=self.pool_size, ignore_border=True, mode='max')

    def activation(self, x):
        return T.tanh(self.max_pool(self.convolve(x)) + self.b.dimshuffle('x', 0, 'x', 'x'))
