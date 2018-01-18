import theano
import theano.tensor as T
import numpy as np
from tools.rng_tools import get_numpy_rng


class ERNN(object):
    """ Elman Recurrent Neural Network Model """
    @property
    def theta(self):
        return [self.Wx, self.Wh, self.Wo]

    @property
    def n_inputs(self):
        return self.Wx.get_value(borrow=True).shape[0]

    @property
    def n_hidden(self):
        return self.Wx.get_value(borrow=True).shape[1]

    @property
    def n_outs(self):
        return self.Wo.get_value(borrow=True).shape[1]

    @property
    def input_shape(self):
        return self.n_inputs,

    def __init__(self, n_inputs, n_hidden, n_out, output_fn=T.nnet.sigmoid, numpy_rng=None):
        numpy_rng = get_numpy_rng(numpy_rng)
        self.h0 = theano.shared(np.zeros(shape=n_hidden, dtype=theano.config.floatX), name='h0')
        self._act_fn = output_fn

        # Weights of input --> hidden
        epsilon = np.sqrt(6. / (n_inputs + n_hidden))
        self.Wx = theano.shared(
            4 * np.asarray(
                numpy_rng.uniform(
                    -epsilon, epsilon,
                    size=(n_inputs, n_hidden)
                ), dtype=theano.config.floatX
            ),
            name='wx'
        )

        # Weights of hidden --> hidden
        epsilon = np.sqrt(6. / (n_hidden + n_hidden))
        self.Wh = theano.shared(
            4 * np.asarray(
                numpy_rng.uniform(
                    -epsilon, epsilon,
                    size=(n_hidden, n_hidden)
                ), dtype=theano.config.floatX,
            ),
            name='wh',
        )

        # Weights of hidden --> output
        epsilon = np.sqrt(6. / (n_hidden + n_out))
        self.Wo = theano.shared(
            4 * np.asarray(
                numpy_rng.uniform(
                    -epsilon, epsilon,
                    size=(n_hidden, n_out)
                ), dtype=theano.config.floatX
            ),
            name='wo'
        )

        self.params = [self.Wx, self.Wh, self.Wo, self.h0]

    def __call__(self, x):
        return self.propagate(x)

    def recurrence_step(self, x, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x, self.Wx) + T.dot(h_tm1, self.Wh))
        out_t = self._act_fn(T.dot(h_tm1, self.Wo))

        return [h_t, out_t]

    def propagate(self, x):
        # Create as many h0 as examples in the batch
        # First use dimshuffle to turn h0 into a row vector
        # Then repeat its value along axis 0 as many times as there are examples in the batch
        h0 = self.h0.dimshuffle('x', 0).repeat(x.shape[0], axis=0)

        # Scan has to iterate through every time step of each test case in x sequentially
        # This means we have to shuffle the first 2 dimension of x (cases and time steps)
        # so that scan processes the values of all cases for each time step at the same time
        x = x.dimshuffle(1, 0, 2)

        [_, s], _ = theano.scan(
            fn=self.recurrence_step,
            sequences=x,
            outputs_info=[h0, None],
            n_steps=x.shape[0]
        )

        # return all but the first time step; relocate axes, first examples then time steps
        return s[1:].dimshuffle(1, 0, 2)


class LSTM(object):
    """ Long Short-Term Memory Model """
    def __init__(self, n_inputs, n_cells, numpy_rng=None):
        self.c0 = theano.shared(np.zeros(shape=n_cells, dtype=theano.config.floatX), 'c0')
        self.h0 = theano.shared(np.zeros(shape=n_cells, dtype=theano.config.floatX), 'h0')

        # Forget Gate
        epsilon = np.sqrt(6. / (n_inputs + 2 * n_cells))
        self.Wf = theano.shared(
            value=4 * numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Wf'
        )

        # Input Gate
        self.Wi = theano.shared(
            value=4 * numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Wr'
        )

        # New values
        self.Nc = theano.shared(
            value=numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Nc'
        )

        # Output Gate
        self.Wo = theano.shared(
            value=4 * numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Wo'
        )

        self.params = [self.c0, self.h0, self.Wf, self.Wo, self.Wi, self.Nc]

    def recurrence_step(self, x, c_tm1, h_tm1):
        inputs = T.concatenate([x, h_tm1], axis=1)

        ft = T.nnet.sigmoid(T.dot(inputs, self.Wf))
        c_update = T.tanh(T.dot(inputs, self.Nc)) * T.nnet.sigmoid(T.dot(inputs, self.Wi))

        ct = ft * c_tm1 + c_update
        output = T.tanh(ct) * T.nnet.sigmoid(inputs, self.Wo)

        return [ct, output]

    def propagate(self, x):
        x = x.dimshuffle(1, 0, 2)

        h0 = self.h0.dimshuffle('x', 0).repeat(x.shape[0], axis=0)
        c0 = self.c0.dimshuffle('x', 0).repeat(x.shape[0], axis=0)

        [_, result], _ = theano.scan(
            fn=self.recurrence_step,
            sequences=x,
            outputs_info=[c0, h0],
            n_steps=x.shape[0]
        )

        return result.dimshuffle(1, 0, 2)


class GRU(object):
    """ Gated Recurrent Unit Model"""
    def __init__(self, n_inputs, n_cells, numpy_rng=None):
        self.c0 = theano.shared(np.zeros(shape=n_cells, dtype=theano.config.floatX), 'c0')

        epsilon = np.sqrt(6. / (n_inputs + 2 * n_cells))

        # Update Gate
        self.Wu = theano.shared(
            value=4 * numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Wr'
        )

        # Output Gate
        self.Nc = theano.shared(
            value=numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Nc'
        )

        # Recurrent Gate
        self.Wr = theano.shared(
            value=4 * numpy_rng.uniform(
                -epsilon, epsilon,
                size=(n_inputs + n_cells, n_cells),
            ),
            name='Wo'
        )

        self.params = [self.c0, self.Wr, self.Wu, self.Nc]

    def recurrence_step(self, x, c_tm1):
        inputs = T.concatenate([x, c_tm1], axis=1)

        zt = T.nnet.sigmoid(T.dot(inputs, self.Wu))
        rt = T.nnet.sigmoid(T.dot(inputs, self.Wr))

        c_update = T.tanh(T.dot(T.concatenate([x, rt * c_tm1], axis=1), self.Wu))

        c_t = (1.0 - zt) * c_tm1 + zt * c_update

        return c_t

    def propagate(self, x):
        x = x.dimshuffle(1, 0, 2)

        c0 = self.c0.dimshuffle('x', 0).repeat(x.shape[0], axis=0)

        result, _ = theano.scan(
            fn=self.recurrence_step,
            sequences=x,
            outputs_info=[c0],
            n_steps=x.shape[0]
        )

        return result.dimshuffle(1, 0, 2)
