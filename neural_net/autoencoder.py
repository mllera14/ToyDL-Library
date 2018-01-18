import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from optimization.cost import CostFunctionBuilder, CostFunction
from collections import OrderedDict
from tools.rng_tools import get_numpy_rng, get_theano_rng


class dA(object):
    """ Denoising autoencoder Model """
    @property
    def theta(self):
        return self.params[0]

    @property
    def bias(self):
        return self.params[1], self.params[2]

    @property
    def input_shape(self):
        return self.params[0].get_value(borrow=True).shape[:1]

    def __init__(self, n_features, code_size, np_rng=None, theano_rng=None, w=None, b=None, b_T=None, act_fn=sigmoid):
        np_rng = get_numpy_rng(np_rng)

        if w is None:
            epsilon = np.sqrt(6. / (code_size + n_features))
            w_values = np.asarray(
                np_rng.uniform(
                    low=-epsilon,
                    high=epsilon,
                    size=(n_features, code_size)
                ), dtype=theano.config.floatX
            )
            if act_fn == T.nnet.sigmoid:
                w_values *= 4.
            w = theano.shared(value=w_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((code_size,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        if b_T is None:
            b_values = np.zeros((n_features,), dtype=theano.config.floatX)
            b_T = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [w, b, b_T]
        self._act_fn = act_fn
        self._theano_rng = get_theano_rng(np_rng if theano_rng is None else theano_rng)

    def __call__(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        return self._act_fn(T.dot(inputs, self.theta) + self.bias[0])

    def decode(self, code):
        return self._act_fn(T.dot(code, self.theta.T) + self.bias[1])

    def propagate(self, inputs):
        return self.encode(inputs)


class SdA(object):
    """ Stacked Denoising autoencoder Model """
    @property
    def theta(self):
        return [layer.theta for layer in self.layers]

    @property
    def bias(self):
        return [layer.bias for layer in self.layers]

    @property
    def feed_forward_params(self):
        return [(da.theta, da.bias[0]) for da in self.layers]

    @property
    def input_shape(self):
        return self.layers[0].params[0].get_value(borrow=True).shape[:1]

    def __call__(self, inputs):
        return self.decode(self.encode(inputs))

    def __init__(self, n_inputs, hidden_dims, np_rng=None, theano_rng=None):
        np_rng, theano_rng = get_numpy_rng(np_rng), get_theano_rng(theano_rng)

        dims = [n_inputs] + hidden_dims

        self.layers, self.params = [], []
        for i in range(1, len(dims)):
            da = dA(
                dims[i - 1], dims[i],
                np_rng, theano_rng,
            )
            self.layers.append(da)
            self.params.extend(da.params)

    def encode(self, inputs):
        x = inputs
        for da in self.layers:
            x = da.encode(x)
        return x

    def decode(self, inputs):
        x = inputs
        for da in self.layers:
            x = da.decode(x)
        return x

    def propagate(self, inputs):
        return self.encode(inputs)


class dAReconstructionError(CostFunctionBuilder):
    def __init__(self, l1=0.0, l2=0.0, corruption_level=0.0, theano_rng=None):
        CostFunctionBuilder.__init__(self, l1, l2)
        self.corruption_level = corruption_level
        self._theano_rng = get_theano_rng(theano_rng)

    def __call__(self, model, data, params=None):
        # noinspection PyShadowingNames
        def denoised_output(inputs):
            corrupted_input = self.get_corrupted_input(inputs, self.corruption_level)
            return model.decode(model.encode(corrupted_input))

        inputs = data.input_tensor_type('inputs')
        reg = self.regularization_term(params)
        output = T.mean(T.sum(T.nnet.binary_crossentropy(denoised_output(inputs), inputs), axis=1))
        return CostFunction(inputs, output + reg, OrderedDict(), 'DenoisingError')

    def get_corrupted_input(self, inputs, corruption_level):
        corruption_model = self._theano_rng.binomial(
            size=inputs.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        )
        return corruption_model * inputs
