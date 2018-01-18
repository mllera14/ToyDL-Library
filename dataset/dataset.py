import theano
import theano.tensor as T
import numpy as np


class Dataset(object):
    def __init__(self, x, y=None, problem_type='classification'):
        # type: (np.ndarray, np.ndarray, str) -> None
        """
        :param x:
        :param y:
        :param problem_type: One of options: classification, regression, unsupervised
        """
        if y is None and problem_type != 'unsupervised':
            raise ValueError('Supervised dataset must have target values')

        self.data = x, y
        self.problem_type = problem_type

    @property
    def size(self):
        return self.inputs.shape[0]

    @property
    def inputs(self):
        return self.data[0]

    @property
    def targets(self):
        return self.data[1]

    @property
    def input_tensor_type(self):
        if len(self.inputs.shape) == 1:
            return T.vector
        if len(self.inputs.shape) == 2:
            return T.matrix
        if len(self.inputs.shape) == 3:
            return T.tensor3
        return T.tensor4

    @property
    def target_tensor_type(self):
        if len(self.targets.shape) == 1:
            return T.ivector if self.problem_type == 'classification' else T.vector

        if len(self.targets.shape) == 2:
            return T.imatrix if self.problem_type == 'classification' else T.matrix

        if len(self.targets.shape) == 3:
            return T.itensor3 if self.problem_type == 'classification' else T.tensor3

        if self.problem_type:
            return T.itensor4 if self.problem_type == 'classification' else T.tensor4

    def shared_dataset(self, borrow=True):
        shared_inputs = theano.shared(self.inputs.astype(dtype=theano.config.floatX), 'inputs', borrow=borrow)
        shared_outputs = theano.shared(self.targets.astype(dtype=theano.config.floatX), borrow=borrow)\
            if self.targets is not None else None

        if self.problem_type == 'classification':
            shared_outputs = T.cast(shared_outputs, dtype='int32')

        return shared_inputs, shared_outputs

    def transform_data(self, f):
        return Dataset(f(self.inputs), f(self.targets) if self.targets is not None else None, self.problem_type)

    def get_unsupervised_dataset(self):
        return Dataset(self.inputs, problem_type='unsupervised')


def split(dataset, ratio):
    if not (0 < ratio < 1):
        raise ValueError('Ratio must be strictly between 0 and 1')

    if isinstance(dataset, np.ndarray):
        x, y = dataset
        n_train = np.int(ratio * x.shape[0])

        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = (y[:n_train], y[n_train:]) if y is not None else (None, None)

        return (x_train, y_train), (x_test, y_test)

    if isinstance(dataset, Dataset):
        x, y = dataset.data

        n_train = np.int(ratio * x.shape[0])

        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = (y[:n_train], y[n_train:]) if y is not None else (None, None)

        return Dataset(x_train, y_train, dataset.problem_type), Dataset(x_test, y_test, dataset.problem_type)

    raise ValueError('Unknown value for parameter dataset')
