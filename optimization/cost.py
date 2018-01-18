import theano
import theano.tensor as T
from collections import OrderedDict
from copy import copy


class CostFunction(object):
    def __init__(self, inputs, output, updates, cost_type, target=None, derivative=None):
        self.input = inputs
        self.target = target
        self.output = output
        self.type = cost_type
        self._updates = updates
        self._derivative = derivative

    @property
    def updates(self):
        return copy(self._updates)

    def derivative(self, params):
        if self._derivative is None:
            return T.grad(self.output, params)
        else:
            return self._derivative(params)


class CostFunctionBuilder(object):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def regularization_term(self, params):
        regularization = 0.0

        if params is not None:
            if self.l1:
                regularization += self.l1 * T.sum(abs(params))
            if self.l2:
                if isinstance(params, list):
                    l2_reg_term = T.sum([T.sum(p ** 2) for p in params])
                else:
                    l2_reg_term = T.sum(params ** 2)
                regularization += self.l2 * l2_reg_term

        return regularization


class MSE(CostFunctionBuilder):
    def __init__(self, l1=0.0, l2=0.0):
        CostFunctionBuilder.__init__(self, l1, l2)

    def __call__(self, fn, data, params=None):
        inputs = data.input_tensor_type('inputs')
        target = data.target_tensor_type('target')

        reg = self.regularization_term(params)

        if data.problem_type == 'Classification':
            output = fn(inputs)[T.arange(target.shape[0]), target]
            output = T.mean(T.sqr(1.0 - output))
            cost = 'Supervised MSE'

        elif data.problem_type == 'Regression':
            output = T.mean(T.sqr(target - fn(inputs)), dtype=theano.config.floatX)
            cost = 'Regression MSE'

        else:
            raise ValueError()

        return CostFunction(inputs=inputs, output=output + reg, updates=OrderedDict(), cost_type=cost, target=target)


class BinXENT(CostFunctionBuilder):
    def __call__(self, fn, data, params=None):
        inputs = data.input_tensor_type('inputs')
        target = data.target_tensor_type('target')
        reg = self.regularization_term(params)

        output = T.mean(
            T.sum(T.nnet.binary_crossentropy(fn(inputs), target), axis=target.ndim - 1), dtype=theano.config.floatX)

        return CostFunction(inputs, output + reg, OrderedDict(), 'BinXENT')


class CatXENT(CostFunctionBuilder):
    def __call__(self, fn, data, params=None):
        inputs = data.input_tensor_type('input')
        category = data.target_tensor_type('target')
        reg = self.regularization_term(params)

        output = T.mean(T.nnet.categorical_crossentropy(fn(inputs), category), dtype=theano.config.floatX)
        return CostFunction(inputs, output + reg, OrderedDict(), 'CatXENT', category)


class NLL(CostFunctionBuilder):
    def __call__(self, fn, data, params=None):
        inputs = data.input_tensor_type('input')
        target = data.target_tensor_type('target') if data.targets is not None else None
        reg = self.regularization_term(params)

        if data.problem_type == 'Classification':
            output = -T.mean(T.log(fn(inputs))[T.arange(target.shape[0]), target])
        elif data.problem_type == 'Unsupervised':
            output = -T.mean(T.log(fn(inputs)))
        else:
            raise ValueError()

        return CostFunction(inputs=inputs, output=output + reg, updates=OrderedDict(), cost_type='NLL', target=target)


class ZeroOneLoss(CostFunctionBuilder):
    def __call__(self, fn, data):
        inputs = data.input_tensor_type('input')
        target = data.target_tensor_type('target')

        output = T.mean(T.neq(target, T.argmax(fn(inputs), axis=1)))
        return CostFunction(inputs, output, OrderedDict(), 'ZeroOneLoss', target)


class CDK(CostFunctionBuilder):
    def __init__(self, l1=0.0, l2=0.0, k=1, persistent=True):
        CostFunctionBuilder.__init__(self, l1, l2)
        self.k = k
        self._persistent = persistent

    def __call__(self, model, data, params=None):
        sample = model.sample

        inputs = data.input_tensor_type('input')

        reg = self.regularization_term(params)

        if not self._persistent:
            chain_state = inputs
        else:
            chain_state = theano.shared(value=data.inputs[:self.k], name='chain_state')

        (v_samples, pre_act), updates = theano.scan(sample, outputs_info=[chain_state, None], n_steps=self.k)
        chain_end = v_samples[-1]

        if not self._persistent:
            cost_type = 'CD{0}'.format(self.k)
            monitoring_fn = T.nnet.binary_crossentropy(T.nnet.sigmoid(pre_act[-1]), inputs)
        else:
            cost_type = 'P-CD{0}'.format(self.k)
            updates[chain_state] = chain_end

            # pseudo (negative) log-likelihood
            n = inputs.shape[1]
            bit_i = theano.shared(value=0, name='bit_i')

            xi = T.round(inputs)
            p_xi = model(xi)

            xi_flip = T.set_subtensor(xi[:, bit_i], 1 - xi[:, bit_i])
            p_xi_flip = model(xi_flip)

            monitoring_fn = -T.mean(n * T.log(T.nnet.sigmoid(p_xi_flip - p_xi)))
            updates[bit_i] = T.cast((bit_i + 1) % n, 'int32')

        updates = updates
        cost = T.mean(model(inputs)) - T.mean(model(chain_end)) + reg

        def derivative(p):
            return T.grad(cost, wrt=p, consider_constant=[chain_end])

        return CostFunction(inputs, monitoring_fn, updates, cost_type, derivative=derivative)
