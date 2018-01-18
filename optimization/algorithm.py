import theano
import theano.tensor as T
import numpy as np
from optimization.cost import CostFunction


class OptimizationMethod(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.gradient_step = None
        self.n_batches = 0

    def __call__(self):
        if self.gradient_step is None:
            raise ValueError('Method not initialized')

        return np.mean([self.gradient_step(i) for i in range(self.n_batches)])

    def initialize(self, j_train, params, training_set):
        if not isinstance(j_train, CostFunction):
            raise ValueError('Invalid type for argument j_train')

        self.gradient_step = self._compile_gradient_step(j_train, params, training_set.shared_dataset(borrow=True))
        self.n_batches = np.int(np.ceil(training_set.size / np.float(self.batch_size)))

    def optimize(self, j_train, params, training_examples):
        return self(j_train, params, training_examples)

    def _compile_gradient_step(self, j_train, params, training_examples):
        raise NotImplementedError()


class SGD(OptimizationMethod):
    """ Stochastic Gradient Descent optimization method """
    def __init__(self, alpha=0.1, batch_size=20):
        OptimizationMethod.__init__(self, batch_size)
        self.init_alpha = alpha
        self.alpha = theano.shared(np.float32(alpha), 'alpha')

    def _compile_gradient_step(self, j_train, params, training_examples):
        print('Compiling SGD gradient descent step function')

        self.alpha.set_value(self.init_alpha)

        index = T.lscalar('index')
        updates = j_train.updates

        nabla_params = j_train.derivative(params)
        for param, gparam in zip(params, nabla_params):
            updates[param] = param - self.alpha * gparam

        training_x, training_y = training_examples

        givens = {j_train.input: training_x[self.batch_size * index: self.batch_size * (index + 1)]}
        if j_train.target is not None:
            givens[j_train.target] = training_y[self.batch_size * index: self.batch_size * (index + 1)]

        gradient_step = theano.function(
            inputs=[index],
            outputs=j_train.output,
            updates=updates,
            givens=givens
        )

        return gradient_step


class Momentum(OptimizationMethod):
    """ SGD with momentum optimization method """
    def __init__(self, alpha=0.01, rho=0.9, batch_size=20):
        OptimizationMethod.__init__(self, batch_size)
        self.init_alpha = alpha
        self.init_rho = rho
        self.alpha = theano.shared(np.float32(self.init_alpha), name='alpha')
        self.rho = theano.shared(np.float32(self.init_rho), name='rho')

    def _compile_gradient_step(self, j_train, params, training_examples):
        print('Compiling Momentum gradient descent step function')

        self.alpha.set_value(self.init_alpha)
        self.rho.set_value(self.init_rho)

        index = T.iscalar('index')
        updates = j_train.updates
        batch_size = self.batch_size
        nabla_params = j_train.derivative(params)

        for p, g in zip(params, nabla_params):
            acc = theano.shared(
                value=np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX), name='velocity vector')
            v_t = self.rho * acc - self.alpha * g
            updates[p] = p + v_t
            updates[acc] = v_t

        training_x, training_y = training_examples

        givens = {j_train.input: training_x[batch_size * index: batch_size * (index + 1)]}
        if j_train.target is not None:
            givens[j_train.target] = training_y[batch_size * index: batch_size * (index + 1)]

        gradient_step = theano.function(
            inputs=[index],
            outputs=j_train.output,
            updates=updates,
            givens=givens
        )

        return gradient_step


class NAG(OptimizationMethod):
    """ Nesterov Accelerated Gradient (SGD + Nesterov momentum) optimization method """
    def __init__(self, alpha=0.01, rho=0.9, batch_size=20):
        OptimizationMethod.__init__(self, batch_size)
        self.init_alpha = alpha
        self.init_rho = rho
        self.alpha = theano.shared(np.float32(self.init_alpha), name='alpha')
        self.rho = theano.shared(np.float32(self.init_rho), name='rho')

    def _compile_gradient_step(self, j_train, params, training_examples):
        print('Compiling NAG gradient descent step function')

        self.alpha.set_value(self.init_alpha)
        self.rho.set_value(self.init_rho)

        index = T.iscalar('index')
        updates = j_train.updates
        batch_size = self.batch_size
        nabla_params = j_train.derivative(params)
        updates_aux = []

        for p, g in zip(params, nabla_params):
            acc = theano.shared(
                np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX), 'accumulated gradient')

            updates_aux.append((p, p - self.alpha * acc))

            updates[p] = p - self.alpha * g
            updates[acc] = (1.0 - self.rho) * acc + self.rho * g

        jump_step = theano.function(
            inputs=[],
            updates=updates_aux
        )

        training_x, training_y = training_examples

        givens = {j_train.input: training_x[batch_size * index: batch_size * (index + 1)]}
        if j_train.target is not None:
            givens[j_train.target] = training_y[batch_size * index: batch_size * (index + 1)]

        correction_step = theano.function(
            inputs=[index],
            outputs=j_train.output,
            updates=updates,
            givens=givens
        )

        def gradient_step(i):
            jump_step()
            return correction_step(i)

        return gradient_step


class RMSprop(OptimizationMethod):
    """ Root Mean Square proportional optimization method """
    def __init__(self, alpha=0.1, rho=0.9, batch_size=20, epsilon=1e-6):
        OptimizationMethod.__init__(self, batch_size)
        self.init_alpha = alpha
        self.init_rho = rho
        self.eps = epsilon
        self.alpha = theano.shared(np.float32(self.init_alpha), name='alpha')
        self.rho = theano.shared(np.float32(self.init_rho), name='rho')

    def _compile_gradient_step(self, j_train, params, training_examples):
        print('Compiling RMSprop gradient descent step function')

        self.alpha.set_value(self.init_alpha)
        self.rho.set_value(self.init_rho)

        index = T.iscalar('index')
        updates = j_train.updates
        batch_size = self.batch_size
        nabla_params = j_train.derivative(params)

        for p, g in zip(params, nabla_params):
            acc = theano.shared(
                value=np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX),
                name='accumulated_gradients'
            )

            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            g_scale = T.sqrt(acc_new + self.eps)
            g /= g_scale

            updates[p] = p - self.alpha * g
            updates[acc] = acc_new

        training_x, training_y = training_examples

        givens = {j_train.input: training_x[batch_size * index: batch_size * (index + 1)]}
        if j_train.target is not None:
            givens[j_train.target] = training_y[batch_size * index: batch_size * (index + 1)]

        gradient_step = theano.function(
            inputs=[index],
            outputs=j_train.output,
            updates=updates,
            givens=givens
        )

        return gradient_step
