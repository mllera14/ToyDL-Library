import numpy as np
import theano
import theano.tensor as T
from tools.rng_tools import get_numpy_rng, get_theano_rng


class RBM(object):
    """ Restricted Boltzmann Machine Model """
    @property
    def theta(self):
        return self.W

    @property
    def bias(self):
        return self.hbias, self.vbias

    @property
    def input_shape(self):
        return self.W.get_value(borrow=True).shape[:1]

    def __init__(self, n_visible, n_hidden, init_w=None, vbias=None, hbias=None, numpy_rng=None, theano_rng=None):
        self.n_visible, self.n_hidden = n_visible, n_hidden

        np_rng, theano_rng = get_numpy_rng(numpy_rng), get_theano_rng(theano_rng)

        if init_w is None:
            w_values = np.asarray(
                numpy_rng.normal(
                    loc=0.0,
                    scale=0.1,
                    size=(n_visible, n_hidden)
                ), dtype=theano.config.floatX
            )
            init_w = theano.shared(value=w_values, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # These are the model parameters:
        # self.W are the weights associated with each edge w_ij where i is the visible unit and j is the hidden one
        # self.vbias and self.hbias are the biases for visible and hidden units respectively, one per row
        # Given this structure, input is expected as a matrix where rows are examples and columns are feature values
        self.W, self.vbias, self.hbias = init_w, vbias, hbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

    def __call__(self, v):
        return self.free_energy(v)

    @property
    def sample(self):
        return self.gibbs_step_given_v

    def prop_up(self, visible):
        pre_act = T.dot(visible, self.W) + self.hbias
        return T.nnet.sigmoid(pre_act), pre_act

    def prop_down(self, hidden):
        pre_act = T.dot(hidden, self.W.T) + self.vbias
        return T.nnet.sigmoid(pre_act), pre_act

    def propagate(self, visible):
        return self.prop_up(visible)[0]

    def down_propagate(self, hidden):
        return self.prop_down(hidden)[0]

    def sample_vgh(self, h_sample):
        act, pre_sigm = self.prop_down(h_sample)

        v_sample = self.theano_rng.binomial(
            size=act.shape,
            n=1, p=act,
            dtype=theano.config.floatX
        )

        return v_sample, pre_sigm

    def sample_hgv(self, v_sample):
        act, pre_sigm = self.prop_up(v_sample)

        h_sample = self.theano_rng.binomial(
            size=act.shape,
            n=1, p=act,
            dtype=theano.config.floatX
        )

        return h_sample, pre_sigm

    def gibbs_step_given_v(self, v_sample):
        h_sample, _ = self.sample_hgv(v_sample)
        return self.sample_vgh(h_sample)

    def gibbs_step_given_h(self, h_sample):
        v_sample, _ = self.sample_vgh(h_sample)
        return self.sample_hgv(v_sample)

    def free_energy(self, v):
        xw_plus_hbias = T.dot(v, self.W) + self.hbias
        xvbias = T.dot(v, self.vbias)
        return -T.sum(T.nnet.softplus(xw_plus_hbias), axis=1) - xvbias

    def prob_h_given_v(self, h, v):
        pre_sig = T.dot(v, self.W) + self.hbias
        return T.prod(T.nnet.sigmoid(T.pow(-1, 1 - h) * pre_sig), axis=1)

    def prob_v_given_h(self, v, h):
        pre_sig = T.dot(h, self.W.T) + self.vbias
        return T.prod(T.nnet.sigmoid(T.pow(-1, 1 - v) * pre_sig), axis=1)


class GaussianRBM(RBM):
    """ Gaussian-Bernoulli RBM Model """
    def __init__(self, n_visible, n_hidden, init_w=None, vbias=None, hbias=None, cov_val=None, fixed_covariance=True,
                 numpy_rng=None, theano_rng=None,):
        RBM.__init__(self, n_visible, n_hidden, init_w, vbias, hbias, numpy_rng, theano_rng)

        if cov_val is None:
            values = np.ones(n_visible, dtype=theano.config.floatX)
        elif isinstance(cov_val, list):
            if len(cov_val) != n_visible:

                raise ValueError('List of covariance values must be equal to the number of visible units')
            values = np.asarray(cov_val, dtype=theano.config.floatX)
        elif isinstance(cov_val, float):
            values = cov_val * np.ones(n_visible, dtype=theano.config.floatX)
        else:
            raise ValueError('Wrong type for cov_val, expected float or list got {0}'.format(type(cov_val)))

        cov_matrix_diag = theano.shared(
            value=values,
            borrow=True,
            name='sigma'
        )

        self.sigma = cov_matrix_diag
        if not fixed_covariance:
            self.params.append(cov_matrix_diag)

    def prop_up(self, visible):
        pre_act = T.dot(visible / self.sigma, self.W) + self.hbias
        return T.nnet.sigmoid(pre_act), pre_act

    def free_energy(self, v):
        squared_term = 0.5 * T.sum(T.sqr((v - self.vbias) / self.sigma), axis=1)
        xw_plus_hbias = T.dot(v / self.sigma, self.W) + self.hbias

        return -T.sum(T.nnet.softplus(xw_plus_hbias), axis=1) - squared_term

    def sample_vgh(self, h_sample):
        mean, pre_sigm = self.prop_down(h_sample)

        v_sample = self.theano_rng.normal(
            size=mean.shape,
            avg=mean,
            std=self.sigma,
            dtype=theano.config.floatX
        )

        return v_sample, pre_sigm


class DBN(object):
    """ Deep Belief Network Model """
    @property
    def theta(self):
        return [layer.theta for layer in self.layers]

    @property
    def bias(self):
        return [layer.bias for layer in self.layers]

    @property
    def feed_forward_params(self):
        return [rbm.params[:-1] for rbm in self.layers]

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    def __init__(self, n_inputs, hidden_info, np_rng=None, theano_rng=None):
        np_rng, theano_rng = get_numpy_rng(np_rng), get_theano_rng(theano_rng)

        dims = [n_inputs] + hidden_info

        layers = []
        for i in range(1, len(dims)):
            rbm = RBM(
                n_visible=dims[i - 1],
                n_hidden=dims[i],
                numpy_rng=np_rng,
                theano_rng=theano_rng
            )
            layers.append(rbm)

        self.layers = layers
        self.numpy_rng = np_rng

    def generate_sample(self, k=100, n_samples=1, init_values=None):
        top_rbm, bottom_rbm = self.layers[-1], self.layers[0]

        if init_values is None:
            init_values = self.numpy_rng.binomial(
                size=(n_samples, top_rbm.n_hidden),
                n=1, p=0.5
            ).astype(dtype=theano.config.floatX, copy=False)

        chain_init = theano.shared(
            value=init_values,
            name='chain_init'
        )

        (v_samples, _), updates = theano.scan(
            fn=top_rbm.gibbs_step_given_h,
            outputs_info=[chain_init, None],
            n_steps=k
        )

        chain_end = v_samples[-1]
        updates[chain_init] = chain_end
        hidden_sample = chain_end

        for rbm in reversed(self.layers[1:]):
            hidden_sample, _ = rbm.sample_vgh(hidden_sample)
        prob = bottom_rbm.down_propagate(hidden_sample)

        return prob, updates
