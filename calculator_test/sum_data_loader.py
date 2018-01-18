import numpy as np
from itertools import product
from dataset.dataset import Dataset
from wrapper.calculator import to_binary
from tools.rng_tools import get_numpy_rng


def create_training_data():
    training_x = []
    training_y = []

    for i, j in product(range(256), repeat=2):
        bin_i = to_binary(i)
        bin_i.extend([0] * (11 - len(bin_i)))

        bin_j = to_binary(j)
        bin_j.extend([0] * (11 - len(bin_j)))

        bin_ipj = to_binary(i + j)
        bin_ipj.extend([0] * (10 - len(bin_ipj)))

        training_x.append(np.transpose([bin_i, bin_j]))
        training_y.append(bin_ipj)

    return Dataset(
        np.asarray(training_x, dtype=np.int32),
        np.asarray(training_y, dtype=np.int32).reshape((len(training_y), 256 * 2, 1)),
        problem_type='Regression'
    )


def create_test_data(rng, low=10000, high=20000, size=1000):
    addends = np.asarray(rng.randint(low, high, size=(size, 2)), dtype=np.int32)

    return Dataset(addends, addends.sum(axis=1))


def load_dataset(rng=None):
    rng = get_numpy_rng(rng)

    training = create_training_data()
    testing = create_test_data(rng)

    return training, testing


