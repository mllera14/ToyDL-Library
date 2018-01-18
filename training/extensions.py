import numpy as np


class Extension(object):
    def __call__(self, info):
        raise NotImplementedError()

    def reset(self, info):
        raise NotImplementedError()


class EarlyStopper(Extension):
    def __init__(self, patience=1000, patience_factor=2, improvement_threshold=0.995):
        self.patience = patience
        self.factor = patience_factor
        self.threshold = improvement_threshold
        self.best_valid_error = np.inf
        self.best_params = None
        self.initial_values = {'patience': patience, 'factor': patience_factor, 'threshold': improvement_threshold}

    def __call__(self, info):
        try:
            valid_error = info.last_iter['validation_error']
        except KeyError:
            raise KeyError('No validation error found, cannot execute early stopping')

        j = info.last_iter['current_epoch']
        params = info['params']

        if valid_error < self.best_valid_error:
            if valid_error < self.threshold * self.best_valid_error:
                self.patience = max(self.patience, j * self.factor)

            self.save_params(params)
            self.best_valid_error = valid_error

        if self.patience < j:
            self.__dict__['best_validation_error'] = self.best_valid_error
            self.load_params(params)
            raise StopIteration()

        if info['epochs'] == j:
            self.load_params(params)
            self.__dict__['best_validation_error'] = self.best_valid_error

    def reset(self, info):
        self.__dict__.update(self.initial_values)
        self.best_valid_error = np.inf
        self.best_params = None
        self.save_params(info['params'])

    def load_params(self, params):
        for p, best_p in zip(params, self.best_params):
            p.set_value(best_p)

    def save_params(self, params):
        self.best_params = [p.get_value() for p in params]


class ProgressTracker(Extension):
    def __init__(self, monitoring_function=None):
        self.monitoring_fn = monitoring_function

    def reset(self, info):
        pass

    def __call__(self, info):
        j = info.last_iter['current_epoch']
        n_epochs = info['epochs']

        try:
            print('Epoch {0} of {1}:\n\tMean batch error={2}'.format(j, n_epochs, info.last_iter['training_error']))
            if 'validation_error' in info.last_iter:
                print('\tValidation error = {0}'.format(info.last_iter['validation_error']))
            if 'monitoring_error' in info.last_iter:
                print('\tMonitoring error = {0}'.format(info.last_iter['monitoring_error']))

        except KeyError:
            pass


class VariableUpdater(Extension):
    def __init__(self, name, stop, interval=1, space='line'):
        if space == 'line':
            space = np.linspace
        elif space == 'log':
            space = np.logspace
        else:
            raise ValueError('Parameter space must be either line or log')

        self.name = name
        self.variable = None
        self.space = space
        self.values = None
        self.stop = stop
        self.interval = interval

    def __call__(self, info):
        epoch = info.last_iter['current_epoch']
        if epoch % self.interval == 0:
            self.variable.set_value(self.values[epoch])

    def reset(self, info):
        self.values = self.space(info['init_' + self.name], self.stop, info['epochs'])
        self.variable = info[self.name]
