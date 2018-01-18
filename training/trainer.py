import theano
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from copy import deepcopy

# import theano.tensor as T
from dataset.dataset import split
from optimization.cost import ZeroOneLoss, MSE, NLL


loss_functions = {
    'regression': MSE(),
    'classification': ZeroOneLoss(),
    'unsupervised': NLL()
}


class TrainingInfo(object):
    def __init__(self, components):
        self.parameters = {str(c): c.__dict__ for c in components}
        self.history = []

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.history[item]

        value = None
        for params in self.parameters.values():
            if item in params:
                return params[item]

        if value is None:
            raise KeyError('Key {0} is not found in TrainingInfo object'.format(item))

    def __setitem__(self, key, value):
        self.parameters[key] = value

    @property
    def last_iter(self):
        return self.history[-1]

    def append(self, info):
        self.history.append(info)

    def training_errors(self):
        return [epoch['training_error'] for epoch in self.history]

    def validation_errors(self):
        if all(['validation_error' in epoch for epoch in self.history]):
            return [epoch['validation_error'] for epoch in self.history]
        return None

    def plot_errors(self):
        plt.plot(np.arange(self.last_iter['current_epoch']) + 1, self.training_errors())
        plt.plot(np.arange(self.last_iter['current_epoch']) + 1, self.validation_errors())
        plt.yscale('log')
        plt.show()


class BaseTrainer(object):
    def __init__(self, algorithm, cost_fn, epochs=100, extensions=None, monitoring_cost_fn=None):
        self.algorithm = algorithm
        self.cost_fn_type = cost_fn
        self.monitoring_fn = monitoring_cost_fn
        self.extensions = extensions
        self.epochs = epochs
        self._training_info = None

    @staticmethod
    def _compile_validation_step(j_validate, validation_examples):
        if j_validate is not None:
            # index = T.lscalar('index')

            validation_x, validation_y = validation_examples.shared_dataset(borrow=True)

            givens = {j_validate.input: validation_x}
            if j_validate.target is not None:
                givens[j_validate.target] = validation_y

            validation_step = theano.function(
                inputs=[],
                outputs=j_validate.output,
                updates=j_validate.updates,
                givens=givens,
            )
        else:
            validation_step = None

        return validation_step

    def initialize(self, model, algorithm, training_set, cost_fn_type, validation_set, monitoring_fn=None):
        self._training_info = TrainingInfo(
            [self, model, algorithm] + (self.extensions if self.extensions is not None else []))

        if validation_set is None:
            training_set, validation_set = split(training_set, ratio=0.8)

        print('Compiling validation and monitoring functions (if any)')

        validation_score = self._compile_validation_step(cost_fn_type(model, validation_set), validation_set)

        if monitoring_fn is not None:
            monitoring_score = self._compile_validation_step(monitoring_fn(model, validation_set), validation_set)
        else:
            monitoring_score = None

        cost_fn = cost_fn_type(model, training_set, model.theta)

        self.algorithm.initialize(cost_fn, model.params, training_set)

        if self.extensions is not None:
            for ext in self.extensions:
                ext.reset(info=self._training_info)

        return validation_score, monitoring_score, training_set, validation_set


class Trainer(BaseTrainer):
    def train(self, model, training_set, validation_set=None, save_file=None, save_interval=1):
        validation_score, monitoring_score, training_set, validation_set = \
            self.initialize(model, self.algorithm, training_set, self.cost_fn_type, validation_set, self.monitoring_fn)

        best_error = np.inf
        best_model = None

        for i in range(self.epochs):
            epoch_info = {'current_epoch': i + 1, 'training_error': self.algorithm(),
                          'validation_error': validation_score()}
            error = epoch_info['validation_error']

            if monitoring_score is not None:
                epoch_info['monitoring_error'] = monitoring_score()

            self._training_info.append(epoch_info)

            if self.extensions is not None:
                try:
                    for e in self.extensions:
                        e(self._training_info)
                except StopIteration:
                    break

            if error < best_error:
                best_error = error
                best_model = deepcopy(model)

            if save_file is not None and (i + 1) % save_interval == 0:
                with open(save_file, 'wb') as f:
                    pkl.dump([self._training_info, best_model], f)

        return self._training_info


class GreedyLayerWiseTrainer(BaseTrainer):
    def train(self, model, training_set, validation_set=None, save_file=None, save_interval=1):
        training_history = []

        for i, layer in enumerate(model.layers):
            if isinstance(self.algorithm, list):
                algorithm = self.algorithm[i]
            else:
                algorithm = self.algorithm

            if isinstance(self.cost_fn_type, list):
                cost_fn_type = self.cost_fn_type[i]
            else:
                cost_fn_type = self.cost_fn_type

            validation_score, monitoring_score, training_set, validation_set = \
                self.initialize(layer, algorithm, training_set, cost_fn_type, validation_set, self.monitoring_fn)

            for j in range(self.epochs):
                epoch_info = {'current_epoch': j + 1, 'training_error': algorithm()}

                if validation_score:
                    epoch_info['validation_error'] = validation_score()

                if self.extensions is not None:
                    for e in self.extensions:
                        e(self._training_info)

                if save_file is not None and (j + 1) % save_interval == 0:
                    with open(save_file, 'wb') as f:
                        pkl.dump([self, model], f)

            training_history.append(self._training_info)

            x = training_set.input_tensor_type('input_layer')
            f = theano.function(
                inputs=[x],
                outputs=layer.propagate(x)
            )

            training_set = training_set.transform_data(f)
            if validation_set is not None:
                validation_set = validation_set.transform_data(f)

        return training_history
